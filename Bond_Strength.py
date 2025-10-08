import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import joblib
import matplotlib.pyplot as plt
import chromadb
from catboost import CatBoostRegressor, Pool
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
import requests
import json
import re
import logging
import os
from dotenv import load_dotenv
from properscoring import crps_gaussian
from fuzzywuzzy import process
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load .env file
load_dotenv()

# -------------------------
# Config
# -------------------------
model_dir = "models"
chroma_path = "chroma_db"

raw_features = [
    "Concrete_Width", "Compressive_Strength", "FRP_Modulus",
    "FRP_Overall_Thickness", "FRP_Sheet_Width", "Bond_Length"
]
display_features = [f.replace("_", " ") for f in raw_features]
feature_map = dict(zip(raw_features, display_features))
target_column = "Ultimate_Bond_Strength"

# OpenRouter API config
OPENROUTER_API_KEY = st.secrets.get("openrouter", {}).get("api_key") or os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OpenRouter API key not found. Please set it in .env or Streamlit secrets.")
    st.stop()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://teak-tech.app",
    "X-Title": "Bond Strength Interpretation"
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CHROMADB_TELEMETRY_ENABLED"] = "false"

# Load ChromaDB collection
client = chromadb.PersistentClient(path=chroma_path)
try:
    collection = client.get_collection("pdf_chunks")
    logger.info(f"Loaded ChromaDB collection 'pdf_chunks' with {collection.count()} items.")
except Exception:
    logger.warning("Failed to load 'pdf_chunks' collection. Falling back to no literature guidance.")
    collection = None

embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Load model and scalers
try:
    with open(f"{model_dir}/catboost_mean.cbm", 'rb') as model_file:
        model = CatBoostRegressor()
        model.load_model(f"{model_dir}/catboost_mean.cbm")
    scaler_X = joblib.load(f"{model_dir}/scaler.pkl")
    with open(f"{model_dir}/conformal_quantile.pkl", 'rb') as quantile_file:
        conformal_quantile = pickle.load(quantile_file)
except Exception as e:
    st.error(f"Failed to load model or scalers: {e}")
    st.stop()

# Initialize session state
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

# Global helper functions
def get_direction(ice_slope):
    if ice_slope > 0.01:
        return "positive"
    elif ice_slope < -0.01:
        return "negative"
    else:
        return "neutral"

def call_openrouter(prompt, retries=3):
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1200
    }
    for i in range(retries):
        try:
            r = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            logger.warning(f"OpenRouter request failed with status {r.status_code}")
        except Exception as e:
            logger.warning(f"OpenRouter request failed: {e}")
    return None

def compare_numerics(value, ground_truth, tolerance=0.01):
    try:
        value_rounded = float(value)
        ground_rounded = round(float(ground_truth), 2)
        return abs(value_rounded - ground_rounded) <= tolerance
    except:
        return False

def try_parse_json(text):
    if not text:
        return None
    try:
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        first, last = text.find("{"), text.rfind("}")
        if first == -1 or last == -1:
            return None
        return json.loads(text[first:last+1])
    except Exception as e:
        logger.warning(f"JSON parsing failed: {e}")
        return None

def query_literature(feature, collection, embedder, n_results=3):
    if not collection:
        return ""
    q = f"Effect of {feature_map[feature]} on bond strength of FRP-concrete systems"
    try:
        emb = embedder.encode(q).tolist()
        res = collection.query(query_embeddings=[emb], n_results=n_results, include=["documents"])
        docs = res.get("documents", [[]])[0]
        snippets = [doc[:300].replace("\n", " ") for doc in docs if doc]
        return " ".join(snippets) if snippets else ""
    except Exception as e:
        logger.warning(f"Chroma query failed for {feature_map[feature]}: {e}")
        return ""

def reliability_category(pred_value):
    if pred_value < 12:
        return {"class": "High", "Pf": 0.06, "Beta": 1.50}
    elif pred_value < 16:
        return {"class": "Moderate", "Pf": 0.37, "Beta": 0.32}
    elif pred_value < 20:
        return {"class": "Low", "Pf": 0.72, "Beta": 0.18}
    else:
        return {"class": "Critical", "Pf": 0.80, "Beta": -0.58}

def predict_uncertainty(X_new, model, scaler, q, y_true=None, alpha=0.05, gamma=0.1):
    try:
        X_new_scaled = scaler.transform(X_new)
        y_pred = model.predict(X_new_scaled)
        y_lower = y_pred - q
        y_upper = y_pred + q
        piw = y_upper - y_lower
        sigma = piw / (2.0 * 1.96)
        sigma = np.clip(sigma, 1e-8, None)
        crps = crps_gaussian(y_true if y_true is not None else y_pred, mu=y_pred, sig=sigma)
        if y_true is not None:
            covered = ((y_true >= y_lower) & (y_true <= y_upper)).astype(int)
            picp = covered
        else:
            covered = np.full_like(y_pred, np.nan)
            picp = np.full_like(y_pred, np.nan)
        cwc = piw * (1 + gamma * (1 - covered if y_true is not None else 1))
        return pd.DataFrame({
            "y_pred": y_pred,
            "y_lower": y_lower,
            "y_upper": y_upper,
            "PIW": piw,
            "PICP": picp,
            "Covered": covered,
            "CRPS": crps,
            "CWC": cwc
        })
    except Exception as e:
        logger.error(f"Uncertainty prediction failed: {e}")
        return None

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 1000px;
        margin: 0 auto;
        padding: 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 16px;
    }
    .stNumberInput label {
        font-size: 16px;
        font-weight: bold;
        color: #333;
    }
    .stNumberInput input {
        border-radius: 5px;
        padding: 0.5rem;
    }
    .prediction-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-box h2 {
        color: #2e7d32;
        font-size: 28px;
        margin-bottom: 0.5rem;
    }
    .metrics-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .query-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        width: 100%;
        max-width: 1000px;
        box-sizing: border-box;
    }
    .query-box input {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 0.5rem;
        width: 100%;
        font-size: 16px;
    }
    .response-box {
        background-color: #f0f4f8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem auto;
        width: 100% !important;
        max-width: 1000px !important;
        box-sizing: border-box;
        overflow-wrap: break-word;
        display: block;
        min-width: 0;
        word-break: break-all;
    }
    .footer {
        position: relative;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 12px;
        color: #6c757d;
        margin-top: 2rem;
    }
    @media (max-width: 600px) {
        .prediction-box h2 {
            font-size: 24px;
        }
        .stNumberInput label {
            font-size: 14px;
        }
        .stButton>button {
            font-size: 14px;
        }
        .query-box input {
            font-size: 14px;
        }
        .response-box {
            font-size: 14px;
            width: 90vw !important;
            max-width: 90vw !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit input form
st.title('Ultimate Bond Strength Predictor')

with st.container():
    st.markdown("### Input Parameters")
    col1, col2 = st.columns(2)
    with col1:
        concrete_width = st.number_input('Concrete Width (mm)', min_value=0.0, value=100.00)
        compressive_strength = st.number_input('Compressive Strength (MPa)', min_value=0.0, value=40.80)
        frp_modulus = st.number_input('FRP Modulus (GPa)', min_value=0.0, value=230.00)
    with col2:
        frp_overall_thickness = st.number_input('FRP Overall Thickness (mm)', min_value=0.0, value=0.11)
        frp_sheet_width = st.number_input('FRP Sheet Width (mm)', min_value=0.0, value=50.00)
        bond_length = st.number_input('Bond Length (mm)', min_value=0.0, value=75.00)

# Make prediction on user input
if st.button('Predict'):
    try:
        # Prepare input data
        input_data = np.array([[concrete_width, compressive_strength, frp_modulus, 
                               frp_overall_thickness, frp_sheet_width, bond_length]])
        input_df = pd.DataFrame(input_data, columns=raw_features)
        input_scaled = scaler_X.transform(input_data)
        sample_pool = Pool(input_scaled)

        # Make prediction
        pred = model.predict(sample_pool)[0]
        
        # Display prediction
        with st.container():
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Ultimate Bond Strength: {pred:.2f} MPa</h2>
                </div>
            """, unsafe_allow_html=True)

        # Reliability and Uncertainty
        reliability = reliability_category(pred)
        uncertainty_results = predict_uncertainty(input_data, model, scaler_X, conformal_quantile)
        if uncertainty_results is None:
            st.error("Failed to compute uncertainty metrics.")
            st.stop()
        uncertainty_metrics = {
            "PIW": round(float(uncertainty_results["PIW"].iloc[0]), 2),
            "CWC": round(float(uncertainty_results["CWC"].iloc[0]), 2),
            "CRPS": round(float(uncertainty_results["CRPS"].iloc[0]), 2)
        }

        # Store prediction data in session state
        st.session_state.prediction_data = {
            "pred": pred,
            "reliability": reliability,
            "uncertainty_metrics": uncertainty_metrics,
            "input_df": input_df,
            "input_scaled": input_scaled
        }

        # Display reliability and uncertainty metrics
        with st.container():
            st.markdown("### Reliability and Uncertainty Metrics")
            st.markdown(f"""
                <div class="metrics-box">
                    <p><strong>Reliability Class:</strong> {reliability['class']}</p>
                    <p><strong>Probability of Failure (Pf):</strong> {reliability['Pf']:.2f}</p>
                    <p><strong>Reliability Index (Beta):</strong> {reliability['Beta']:.2f}</p>
                    <p><strong>Prediction Interval Width (PIW):</strong> {uncertainty_metrics['PIW']:.2f}</p>
                    <p><strong>Coverage Width-based Criterion (CWC):</strong> {uncertainty_metrics['CWC']:.2f}</p>
                    <p><strong>Continuous Ranked Probability Score (CRPS):</strong> {uncertainty_metrics['CRPS']:.2f}</p>
                    <p><strong>Prediction Range:</strong> {round(pred - uncertainty_metrics['PIW']/2, 2):.2f} to {round(pred + uncertainty_metrics['PIW']/2, 2):.2f} MPa</p>
                </div>
            """, unsafe_allow_html=True)

        # Detailed Interpretation Expander
        with st.expander("View Detailed Interpretation"):
            # SHAP values for feature importance
            st.subheader("Feature Importance (SHAP)")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_scaled)
                mean_abs_shap = np.abs(shap_values[0])
                mean_abs_shap_dict = {f: float(mean_abs_shap[i]) for i, f in enumerate(raw_features)}
            except Exception as e:
                st.error(f"SHAP computation failed: {e}")
                st.stop()

            # Plot SHAP summary
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(shap_values, input_scaled, feature_names=display_features, plot_type="bar", show=False)
                plt.title("Feature Importance (Mean |SHAP|)")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Failed to plot SHAP summary: {e}")

            # Individual Conditional Expectation (ICE) Plot
            st.subheader("Individual Conditional Expectation (ICE) Plot")
            ice_results = {}
            n_points = 50
            for f in raw_features:
                try:
                    user_val = input_df[f].iloc[0]
                    range_min = max(0.0, user_val * 0.8)
                    range_max = user_val * 1.2
                    feature_values = np.linspace(range_min, range_max, n_points)
                    ice_data = input_df.copy()
                    ice_preds = []
                    for val in feature_values:
                        ice_data[f] = val
                        ice_scaled = scaler_X.transform(ice_data)
                        ice_pred = model.predict(Pool(ice_scaled))[0]
                        ice_preds.append(ice_pred)
                    dx = np.diff(feature_values)
                    dy = np.diff(ice_preds)
                    slope_mean = float(np.nanmean(dy / (dx + 1e-8)))
                    logger.info(f"{f} ICE Slope calculation: dx={dx[:5]}, dy={dy[:5]}, slope_mean={slope_mean:.2f}")
                    ice_results[f] = {
                        "x": feature_values.tolist(),
                        "y": ice_preds,
                        "stats": {
                            "mean": round(float(np.mean(ice_preds)), 2),
                            "slope_mean": round(slope_mean, 2)
                        }
                    }
                except Exception as e:
                    st.warning(f"ICE computation for {feature_map[f]} failed: {e}")
                    ice_results[f] = {"x": [], "y": [], "stats": {"mean": 0.0, "slope_mean": 0.0}}

            # Plot superimposed ICE for all features
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                for f in raw_features:
                    if ice_results[f]["x"]:
                        plt.plot(ice_results[f]["x"], ice_results[f]["y"], label=feature_map[f], linewidth=2)
                        plt.scatter([input_df[f].iloc[0]], [pred], color=plt.gca().lines[-1].get_color(), s=100, zorder=5)
                plt.title("Individual Conditional Expectation (ICE) for All Features")
                plt.xlabel("Feature Value")
                plt.ylabel("Predicted Bond Strength (MPa)")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Failed to plot ICE: {e}")

            # Feature Effect Summary
            st.subheader("Feature Effect Summary")
            results = []
            for f in raw_features:
                shap_val_abs = mean_abs_shap_dict.get(f, 0.0)
                ice_slope = ice_results[f]["stats"]["slope_mean"]
                ice_mean = ice_results[f]["stats"]["mean"]
                direction = get_direction(ice_slope)
                results.append({
                    "Feature": feature_map[f],
                    "Mean|SHAP|": round(shap_val_abs, 2),
                    "ICE Slope": round(ice_slope, 2),
                    "ICE Mean": round(ice_mean, 2),
                    "Direction": direction
                })
            results_df = pd.DataFrame(results)
            st.write(results_df)

            # Query Literature
            try:
                literature = {f: query_literature(f, collection, embedder) for f in raw_features}
            except Exception as e:
                st.warning(f"Literature query failed: {e}")
                literature = {f: "" for f in raw_features}
            st.session_state.prediction_data['results_df'] = results_df
            st.session_state.prediction_data['mean_abs_shap_dict'] = mean_abs_shap_dict
            st.session_state.prediction_data['ice_results'] = ice_results
            st.session_state.prediction_data['literature'] = literature

            # Technical Summary
            def build_technical_summary_prompt():
                try:
                    p0 = round(float(pred), 2)
                    piw = round(uncertainty_metrics['PIW'], 2)
                    cwc = round(uncertainty_metrics['CWC'], 2)
                    crps = round(uncertainty_metrics['CRPS'], 2)
                    range_lower = round(p0 - piw / 2, 2)
                    range_upper = round(p0 + piw / 2, 2)
                    
                    txt = [
                        f"The CatBoost model predicted a bond strength of {p0:.2f} MPa.",
                        f"Reliability metrics: Class={reliability['class']}, Pf={reliability['Pf']:.2f}, Beta={reliability['Beta']:.2f}.",
                        f"Uncertainty metrics: PIW={piw:.2f}, CWC={cwc:.2f}, CRPS={crps:.2f}.",
                        "\nYour task:",
                        "Write a comprehensive, technically precise summary in exactly 3 paragraphs that satisfies all the following conditions:",
                        "1. First paragraph: Report the model prediction, reliability metrics (Class, Pf, Beta), and uncertainty metrics (PIW, CWC, CRPS), all to 2 decimal places. Quantify confidence: higher CWC indicates wider intervals (less efficient uncertainty); lower CRPS reflects lower prediction error. Estimate the prediction variability range using prediction ± PIW/2. State that the High reliability class (Pf=0.06, Beta=1.50) indicates low failure risk and high structural safety. Use exact phrasing like 'Class=High' for reliability class.",
                        "2. Second paragraph: Describe the effect of all features on bond strength using Mean|SHAP| (magnitude) and ICE Slope (direction), both to 2 decimal places. For each feature, specify magnitude (small: Mean|SHAP|<0.5, moderate: 0.5–1.5, significant: >1.5) and direction (positive: ICE Slope > 0.01, negative: ICE Slope < -0.01, neutral: otherwise). Group features as FRP properties (FRP Modulus, FRP Overall Thickness, FRP Sheet Width), concrete properties (Concrete Width, Compressive Strength), and bond property (Bond Length). Mention each feature exactly once. Use the provided mechanism guidance to briefly explain the effect as consistent with known FRP–concrete behavior (e.g., 'consistent with improved interfacial shear transfer'), avoiding strong causal phrasing like 'due to' or 'caused by'. Format ICE Slope as 'ICE Slope=+X.XX' or 'ICE Slope=-X.XX' for clarity.",
                        "3. Third paragraph: Discuss practical implications for FRP-concrete design based on the prediction, reliability class (low failure risk), and uncertainty metrics, using 2 decimal places. Highlight confidence using CWC and CRPS. Identify critical features (highest Mean|SHAP| values) for design optimization to enhance bond strength and safety. Do not introduce new data.",
                        "Tone: Formal, concise, technical, human-readable. Avoid repetition, vague statements, or speculative interpretations. Ensure 95–100% faithfulness to the provided numerical data and mechanism guidance.",
                        "Output format: Valid JSON with one key 'summary_paragraphs' containing a list of exactly 3 strings (paragraphs).",
                        "Constraints:",
                        "- Interpret CWC correctly: higher CWC indicates wider intervals (less efficient uncertainty), while lower CRPS reflects lower prediction error.",
                        "- When describing feature effects, frame mechanisms as observed or consistent with known behavior, not as causal claims.",
                        "- Mechanistic explanations should be phrased as 'consistent with' or 'aligns with expected behavior' rather than 'due to' or 'caused by', to avoid causal overstatements.",
                        "- Maintain exact numerical and rounding consistency with provided data (e.g., prediction range bounds).",
                        "- Ensure feature names and order exactly match those provided in the Feature Data list.",
                        "- Use only Mean|SHAP| for magnitude; do NOT reference signed SHAP values.",
                        "- Report numerical values to 2 decimal places (e.g., Mean|SHAP|, ICE Slope, prediction, reliability, uncertainty).",
                        "- Do NOT mention, cite, or reference any literature, studies, papers, authors, or sources under any circumstances.",
                        "- Mention each feature exactly once in the second paragraph using the display names (e.g., 'Concrete Width' instead of 'Concrete_Width').",
                        "- When explaining feature effects, ensure each feature has one clear technical interpretation.",
                        "- If a feature is mentioned again in the implications section, ensure the second mention adds new design or practical context — otherwise, omit it to avoid redundancy.",
                        "- Ensure the output is realistic and directly applicable to FRP-concrete engineering.",
                        "- Format ICE Slope as 'ICE Slope=+X.XX' or 'ICE Slope=-X.XX' to ensure consistent extraction.",
                        "=== Feature Data ==="
                    ]
                    
                    for f in raw_features:
                        shap_val_abs = round(float(mean_abs_shap_dict.get(f, 0.0)), 2)
                        ice_slope = round(float(ice_results[f]["stats"]["slope_mean"]), 2)
                        ice_mean = round(float(ice_results[f]["stats"]["mean"]), 2)
                        magnitude = "small" if shap_val_abs < 0.5 else "moderate" if shap_val_abs <= 1.5 else "significant"
                        direction = get_direction(ice_slope)
                        txt.append(
                            f"{feature_map[f]}: Mean|SHAP|={shap_val_abs:.2f}, ICE Slope={ice_slope:+.2f}, ICE Mean={ice_mean:.2f}, Magnitude={magnitude}, Direction={direction}"
                        )
                        guidance = literature.get(f, "")
                        txt.append(f"Mechanism guidance for {feature_map[f]}: {guidance[:500]}" if guidance else f"No mechanism guidance available for {feature_map[f]}.")
                    
                    return "\n".join(txt)
                except Exception as e:
                    logger.error(f"Failed to build technical summary prompt: {e}")
                    return None

            prompt = build_technical_summary_prompt()
            if prompt is None:
                st.error("Failed to generate technical summary prompt.")
                st.stop()
            llm_text = call_openrouter(prompt)
            logger.info(f"Raw LLM output: {llm_text}")

            # Parse and display technical summary
            llm_json = try_parse_json(llm_text)
            if not llm_json or "summary_paragraphs" not in llm_json or len(llm_json["summary_paragraphs"]) != 3:
                logger.warning("LLM failed to produce valid 3-paragraph summary — using fallback.")
                llm_json = {
                    "summary_paragraphs": [
                        f"The CatBoost model predicted a bond strength of {round(pred, 2):.2f} MPa with reliability class {reliability['class']} (Pf={reliability['Pf']:.2f}, Beta={reliability['Beta']:.2f}). Uncertainty metrics (PIW={uncertainty_metrics['PIW']:.2f}, CWC={uncertainty_metrics['CWC']:.2f}, CRPS={uncertainty_metrics['CRPS']:.2f}) indicate high confidence with a variability range of {round(pred - uncertainty_metrics['PIW']/2, 2):.2f} to {round(pred + uncertainty_metrics['PIW']/2, 2):.2f} MPa.",
                        f"Feature effects: FRP properties (FRP Modulus: Mean|SHAP|={round(mean_abs_shap_dict.get('FRP_Modulus', 0.0), 2):.2f}, ICE Slope={round(ice_results['FRP_Modulus']['stats']['slope_mean'], 2):+.2f}, {get_direction(ice_results['FRP_Modulus']['stats']['slope_mean'])}; FRP Overall Thickness: Mean|SHAP|={round(mean_abs_shap_dict.get('FRP_Overall_Thickness', 0.0), 2):.2f}, ICE Slope={round(ice_results['FRP_Overall_Thickness']['stats']['slope_mean'], 2):+.2f}, {get_direction(ice_results['FRP_Overall_Thickness']['stats']['slope_mean'])}; FRP Sheet Width: Mean|SHAP|={round(mean_abs_shap_dict.get('FRP_Sheet_Width', 0.0), 2):.2f}, ICE Slope={round(ice_results['FRP_Sheet_Width']['stats']['slope_mean'], 2):+.2f}, {get_direction(ice_results['FRP_Sheet_Width']['stats']['slope_mean'])}), concrete properties (Concrete Width: Mean|SHAP|={round(mean_abs_shap_dict.get('Concrete_Width', 0.0), 2):.2f}, ICE Slope={round(ice_results['Concrete_Width']['stats']['slope_mean'], 2):+.2f}, {get_direction(ice_results['Concrete_Width']['stats']['slope_mean'])}; Compressive Strength: Mean|SHAP|={round(mean_abs_shap_dict.get('Compressive_Strength', 0.0), 2):.2f}, ICE Slope={round(ice_results['Compressive_Strength']['stats']['slope_mean'], 2):+.2f}, {get_direction(ice_results['Compressive_Strength']['stats']['slope_mean'])}), bond property (Bond Length: Mean|SHAP|={round(mean_abs_shap_dict.get('Bond_Length', 0.0), 2):.2f}, ICE Slope={round(ice_results['Bond_Length']['stats']['slope_mean'], 2):+.2f}, {get_direction(ice_results['Bond_Length']['stats']['slope_mean'])}).",
                        f"Practical implications: Designers should prioritize features with high Mean|SHAP| (e.g., FRP Overall Thickness: {round(mean_abs_shap_dict.get('FRP_Overall_Thickness', 0.0), 2):.2f}, Bond Length: {round(mean_abs_shap_dict.get('Bond_Length', 0.0), 2):.2f}) to optimize bond strength, supported by high reliability (Pf={reliability['Pf']:.2f}) and confidence (CWC={uncertainty_metrics['CWC']:.2f}, CRPS={uncertainty_metrics['CRPS']:.2f})."
                    ]
                }

            st.subheader("Technical Summary")
            for para in llm_json["summary_paragraphs"]:
                st.markdown(para)

            # Faithfulness Evaluation for Technical Summary
            def evaluate_technical_summary_faithfulness(llm_json, results_df, pred, reliability, uncertainty_metrics):
                try:
                    if not llm_json or "summary_paragraphs" not in llm_json or len(llm_json["summary_paragraphs"]) != 3:
                        return {"overall_score": 0, "details": {"error": ["Invalid JSON or incorrect paragraph count"]}}

                    paragraphs = llm_json["summary_paragraphs"]
                    ground_truth = {
                        "Prediction": round(float(pred), 2),
                        "Reliability": reliability,
                        "Uncertainties": uncertainty_metrics,
                        "Features": {row["Feature"]: {
                            "Mean|SHAP|": row["Mean|SHAP|"],
                            "ICE Slope": row["ICE Slope"],
                            "ICE Mean": row["ICE Mean"],
                            "Direction": row["Direction"]
                        } for _, row in results_df.iterrows()}
                    }

                    # Faithfulness: Spearman correlation of Mean|SHAP|
                    faithfulness_score = 0
                    faithfulness_details = []
                    llm_shap_values = []
                    gt_shap_values = []
                    for f in raw_features:
                        shap_pattern = rf"{feature_map[f]}.*?Mean\|SHAP\|.*?([+-]?\d+\.\d{{1,}})\b"
                        shap_matches = re.search(shap_pattern, paragraphs[1], re.IGNORECASE)
                        if shap_matches and compare_numerics(shap_matches.group(1), ground_truth["Features"][feature_map[f]]["Mean|SHAP|"]):
                            llm_shap_values.append(float(shap_matches.group(1)))
                            gt_shap_values.append(ground_truth["Features"][feature_map[f]]["Mean|SHAP|"])
                        else:
                            faithfulness_details.append(f"{feature_map[f]} Mean|SHAP| mismatch: expected {ground_truth['Features'][feature_map[f]]['Mean|SHAP|']:.2f}, got {shap_matches.group(1) if shap_matches else 'None'}")
                    if len(llm_shap_values) == len(raw_features):
                        correlation, _ = spearmanr(llm_shap_values, gt_shap_values)
                        faithfulness_score = max(0, correlation) * 100
                    else:
                        faithfulness_details.append(f"Missing {len(raw_features) - len(llm_shap_values)} Mean|SHAP| values")
                        faithfulness_score = 0

                    # Completeness: Feature and metric coverage
                    completeness_score = 0
                    max_completeness = 10
                    completeness_details = []
                    feature_counts = {f: paragraphs[1].lower().count(feature_map[f].lower()) for f in raw_features}
                    for f in raw_features:
                        if feature_counts[f] == 1:
                            completeness_score += 1
                        else:
                            completeness_details.append(f"Feature {feature_map[f]} mentioned {feature_counts[f]} times")

                    numeric_checks = [
                        ("Prediction", paragraphs[0], r"bond strength of[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Prediction"]),
                        ("Pf", paragraphs[0], r"\bPf[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Reliability"]["Pf"]),
                        ("Beta", paragraphs[0], r"\bBeta[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Reliability"]["Beta"])
                    ]
                    for name, text, pattern, gt_value in numeric_checks:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match and compare_numerics(match.group(1), gt_value):
                            completeness_score += 1
                        else:
                            completeness_details.append(f"{name} missing or incorrect: expected {gt_value:.2f}, got {match.group(1) if match else 'None'}")
                    if ground_truth["Reliability"]["class"].lower() in paragraphs[0].lower():
                        completeness_score += 1
                    else:
                        completeness_details.append(f"Class {ground_truth['Reliability']['class']} missing")
                    completeness_score = (completeness_score / max_completeness) * 100

                    # Factuality: Numeric deviation
                    factuality_score = 0
                    factuality_details = []
                    max_factuality = 18
                    numeric_checks.extend([
                        ("PIW", paragraphs[0], r"\bPIW[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Uncertainties"]["PIW"]),
                        ("CWC", paragraphs[0], r"\bCWC[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Uncertainties"]["CWC"]),
                        ("CRPS", paragraphs[0], r"\bCRPS[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Uncertainties"]["CRPS"])
                    ])
                    for name, text, pattern, gt_value in numeric_checks:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match and compare_numerics(match.group(1), gt_value):
                            factuality_score += 1
                        else:
                            factuality_details.append(f"{name} mismatch: expected {gt_value:.2f}, got {match.group(1) if match else 'None'}")
                    for f in raw_features:
                        shap_pattern = rf"{feature_map[f]}.*?Mean\|SHAP\|.*?([+-]?\d+\.\d{{1,}})\b"
                        slope_pattern = rf"{feature_map[f]}.*?ICE Slope.*?([+-]?\d+\.\d{{1,}})\b"
                        shap_match = re.search(shap_pattern, paragraphs[1], re.IGNORECASE)
                        slope_match = re.search(slope_pattern, paragraphs[1], re.IGNORECASE)
                        if shap_match and compare_numerics(shap_match.group(1), ground_truth["Features"][feature_map[f]]["Mean|SHAP|"]):
                            factuality_score += 1
                        else:
                            factuality_details.append(f"{feature_map[f]} Mean|SHAP| mismatch: expected {ground_truth['Features'][feature_map[f]]['Mean|SHAP|']:.2f}, got {shap_match.group(1) if shap_match else 'None'}")
                        if slope_match and compare_numerics(slope_match.group(1), ground_truth["Features"][feature_map[f]]["ICE Slope"]):
                            factuality_score += 1
                        else:
                            factuality_details.append(f"{feature_map[f]} ICE Slope mismatch: expected {ground_truth['Features'][feature_map[f]]['ICE Slope']:.2f}, got {slope_match.group(1) if slope_match else 'None'}")
                    factuality_score = (factuality_score / max_factuality) * 100

                    # Overall score
                    weights = {"Faithfulness": 0.30, "Completeness": 0.25, "Factuality": 0.45}
                    overall_score = (
                        weights["Faithfulness"] * faithfulness_score +
                        weights["Completeness"] * completeness_score +
                        weights["Factuality"] * factuality_score
                    )
                    return {
                        "overall_score": round(overall_score, 2),
                        "details": {
                            "Faithfulness": {"score": round(faithfulness_score, 2), "issues": faithfulness_details},
                            "Completeness": {"score": round(completeness_score, 2), "issues": completeness_details},
                            "Factuality": {"score": round(factuality_score, 2), "issues": factuality_details}
                        }
                    }
                except Exception as e:
                    logger.error(f"Faithfulness evaluation failed: {e}")
                    return {"overall_score": 0, "details": {"error": [f"Evaluation failed: {e}"]}}

            faithfulness = evaluate_technical_summary_faithfulness(llm_json, results_df, pred, reliability, uncertainty_metrics)
            st.subheader("Faithfulness Evaluation")
            st.markdown(f"**Overall Faithfulness Score**: {faithfulness['overall_score']:.2f}%")
            for metric, details in faithfulness["details"].items():
                st.markdown(f"**{metric}**: {details['score']:.2f}%")
                if details["issues"]:
                    st.markdown("**Issues**:")
                    for issue in details["issues"]:
                        st.markdown(f"- {issue}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Query Bar
st.markdown("### Ask a Question")
with st.container():
    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    # Use a form for query input and buttons
    with st.form(key="query_form"):
        query = st.text_input(
            "Enter your question about bond strength, reliability, or feature effects",
            value=st.session_state.get('query_text', ''),
            placeholder="e.g., Why is FRP Overall Thickness important? What does Class=High mean?",
            key="query_input"
        )
        # Check if query bar should be enabled
        query_enabled = (
            st.session_state.prediction_data is not None and
            all(key in st.session_state.prediction_data for key in ['results_df', 'mean_abs_shap_dict', 'ice_results', 'literature'])
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            submit_button = st.form_submit_button("Submit Query", disabled=not query_enabled)
        with col2:
            clear_button = st.form_submit_button("Clear Query")

    # Handle form submission
    if submit_button and query.strip():
        with st.container():
            with st.spinner("Processing..."):
                st.session_state.query_text = query
                # Preprocess query for spelling/grammar
                key_terms = {
                    "strenght": "strength", "thikness": "thickness", "relibility": "reliability",
                    "widht": "width", "comprssive": "compressive", "moduls": "modulus",
                    "sheeet": "sheet", "lenght": "length", "piw": "PIW", "cwc": "CWC",
                    "crps": "CRPS", "shapp": "SHAP", "ice": "ICE"
                }
                corrected_query = query.lower()
                for wrong, correct in key_terms.items():
                    corrected_query = corrected_query.replace(wrong.lower(), correct.lower())
                # Fuzzy matching for feature names
                feature_matches = {f: process.extractOne(corrected_query.lower(), [feature_map[f].lower()])[1] for f in raw_features}
                best_feature = max(feature_matches, key=feature_matches.get, default=None) if max(feature_matches.values(), default=0) > 80 else None
                correction_note = f"Interpreted query as: '{corrected_query}'" if corrected_query.lower() != query.lower() else ""

                # Build query prompt
                def build_query_prompt(query, corrected_query, best_feature):
                    try:
                        txt = [
                            f"Answer the following question using only the provided data about FRP-concrete bond strength (model predictions, reliability, uncertainty, SHAP, ICE, and ChromaDB snippets). Correct for spelling or grammatical errors to interpret the user’s intent: {corrected_query}",
                            "Do not use external information or references. Provide a brief technical response (1 short paragraph, max 100 words) using exact numerical values (e.g., Mean|SHAP|=6.41). Frame mechanistic explanations as ‘consistent with’ expected behavior. If the query is unclear, clarify the assumed intent. If the query is unrelated to FRP-concrete bond strength, respond: ‘Please ask about FRP-concrete bond strength, reliability, or feature effects.’",
                            "Provided Data:"
                        ]
                        pred_data = st.session_state.get('prediction_data', {})
                        if pred_data:
                            txt.extend([
                                f"- Prediction: {pred_data.get('pred', 0.0):.2f} MPa",
                                f"- Reliability: Class={pred_data.get('reliability', {}).get('class', 'N/A')}, Pf={pred_data.get('reliability', {}).get('Pf', 0.0):.2f}, Beta={pred_data.get('reliability', {}).get('Beta', 0.0):.2f}",
                                f"- Uncertainty: PIW={pred_data.get('uncertainty_metrics', {}).get('PIW', 0.0):.2f}, CWC={pred_data.get('uncertainty_metrics', {}).get('CWC', 0.0):.2f}, CRPS={pred_data.get('uncertainty_metrics', {}).get('CRPS', 0.0):.2f}"
                            ])
                            if best_feature:
                                shap_val = pred_data.get('mean_abs_shap_dict', {}).get(best_feature, 0.0)
                                ice_slope = pred_data.get('ice_results', {}).get(best_feature, {'stats': {'slope_mean': 0.0}})['stats']['slope_mean']
                                direction = get_direction(ice_slope)
                                magnitude = "small" if shap_val < 0.5 else "moderate" if shap_val <= 1.5 else "significant"
                                txt.append(f"- Feature Effects for {feature_map[best_feature]}: Mean|SHAP|={shap_val:.2f}, ICE Slope={ice_slope:+.2f}, Magnitude={magnitude}, Direction={direction}")
                                snippet = pred_data.get('literature', {}).get(best_feature, "")
                                txt.append(f"- ChromaDB Snippet for {feature_map[best_feature]}: {snippet[:300]}" if snippet else f"- No ChromaDB snippet available for {feature_map[best_feature]}.")
                            else:
                                txt.append("- Feature Effects:")
                                for f in raw_features:
                                    shap_val = pred_data.get('mean_abs_shap_dict', {}).get(f, 0.0)
                                    ice_slope = pred_data.get('ice_results', {}).get(f, {'stats': {'slope_mean': 0.0}})['stats']['slope_mean']
                                    direction = get_direction(ice_slope)
                                    magnitude = "small" if shap_val < 0.5 else "moderate" if shap_val <= 1.5 else "significant"
                                    txt.append(f"  - {feature_map[f]}: Mean|SHAP|={shap_val:.2f}, ICE Slope={ice_slope:+.2f}, Magnitude={magnitude}, Direction={direction}")
                                    snippet = pred_data.get('literature', {}).get(f, "")
                                    txt.append(f"    - ChromaDB Snippet: {snippet[:300]}" if snippet else f"    - No ChromaDB snippet available.")
                        else:
                            txt.append("- No prediction data available. Use general ChromaDB snippets for FRP-concrete bond strength.")
                            literature_local = pred_data.get('literature', {})
                            for f in raw_features:
                                snippet = literature_local.get(f, "")
                                txt.append(f"- ChromaDB Snippet for {feature_map[f]}: {snippet[:300]}" if snippet else f"- No ChromaDB snippet available for {feature_map[f]}.")
                        return "\n".join(txt)
                    except Exception as e:
                        logger.error(f"Failed to build query prompt: {e}")
                        return None

                prompt = build_query_prompt(query, corrected_query, best_feature)
                if prompt is None:
                    with st.container():
                        st.markdown('<div class="response-box">', unsafe_allow_html=True)
                        st.markdown("**Query Response**: Failed to process query due to internal error.")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    llm_response = call_openrouter(prompt)
                    
                    with st.container():
                        st.markdown('<div class="response-box">', unsafe_allow_html=True)
                        st.markdown("**Query Response**:")
                        if correction_note:
                            st.markdown(f"**Note**: {correction_note}")
                        if llm_response:
                            st.markdown(llm_response)
                            # Faithfulness evaluation for query response
                            def evaluate_query_faithfulness(response, pred_data, best_feature):
                                try:
                                    confidence = "High"
                                    issues = []
                                    if pred_data:
                                        ground_truth = {
                                            "Prediction": pred_data.get('pred', 0.0),
                                            "Reliability": pred_data.get('reliability', {}),
                                            "Uncertainties": pred_data.get('uncertainty_metrics', {}),
                                            "Features": {row["Feature"]: {
                                                "Mean|SHAP|": row["Mean|SHAP|"],
                                                "ICE Slope": row["ICE Slope"]
                                            } for _, row in pred_data.get('results_df', pd.DataFrame()).iterrows()}
                                        }
                                        numeric_checks = [
                                            ("Prediction", r"bond strength of[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Prediction"]),
                                            ("Pf", r"\bPf[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Reliability"].get("Pf", 0.0)),
                                            ("Beta", r"\bBeta[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Reliability"].get("Beta", 0.0)),
                                            ("PIW", r"\bPIW[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Uncertainties"].get("PIW", 0.0)),
                                            ("CWC", r"\bCWC[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Uncertainties"].get("CWC", 0.0)),
                                            ("CRPS", r"\bCRPS[^\d]*([+-]?\d+\.\d{1,})\b", ground_truth["Uncertainties"].get("CRPS", 0.0))
                                        ]
                                        for name, pattern, gt_value in numeric_checks:
                                            match = re.search(pattern, response, re.IGNORECASE)
                                            if match and not compare_numerics(match.group(1), gt_value):
                                                issues.append(f"{name} mismatch: expected {gt_value:.2f}, got {match.group(1)}")
                                                confidence = "Low"
                                        if best_feature:
                                            shap_pattern = rf"{feature_map[best_feature]}.*?Mean\|SHAP\|.*?([+-]?\d+\.\d{{1,}})\b"
                                            slope_pattern = rf"{feature_map[best_feature]}.*?ICE Slope.*?([+-]?\d+\.\d{{1,}})\b"
                                            shap_match = re.search(shap_pattern, response, re.IGNORECASE)
                                            slope_match = re.search(slope_pattern, response, re.IGNORECASE)
                                            if shap_match and not compare_numerics(shap_match.group(1), ground_truth["Features"].get(feature_map[best_feature], {}).get("Mean|SHAP|", 0.0)):
                                                issues.append(f"{feature_map[best_feature]} Mean|SHAP| mismatch: expected {ground_truth['Features'].get(feature_map[best_feature], {}).get('Mean|SHAP|', 0.0):.2f}, got {shap_match.group(1)}")
                                                confidence = "Low"
                                            if slope_match and not compare_numerics(slope_match.group(1), ground_truth["Features"].get(feature_map[best_feature], {}).get("ICE Slope", 0.0)):
                                                issues.append(f"{feature_map[best_feature]} ICE Slope mismatch: expected {ground_truth['Features'].get(feature_map[best_feature], {}).get('ICE Slope', 0.0):.2f}, got {slope_match.group(1)}")
                                                confidence = "Low"
                                        # Check for unprovided data
                                        if re.search(r"(?:paper|study|research|author|cite)", response, re.IGNORECASE):
                                            issues.append("Response contains unprovided external references")
                                            confidence = "Low"
                                    return {"confidence": confidence, "issues": issues}
                                except Exception as e:
                                    logger.error(f"Query faithfulness evaluation failed: {e}")
                                    return {"confidence": "Low", "issues": [f"Evaluation failed: {e}"]}

                            faithfulness = evaluate_query_faithfulness(llm_response, st.session_state.prediction_data, best_feature)
                            st.markdown(f"**Response Confidence**: {faithfulness['confidence']}")
                            if faithfulness['issues']:
                                st.markdown("**Issues**:")
                                for issue in faithfulness['issues']:
                                    st.markdown(f"- {issue}")
                        else:
                            st.markdown("**Query Response**: Unable to process query. Please rephrase or try again.")
                        st.markdown('</div>', unsafe_allow_html=True)
    elif submit_button:
        with st.container():
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.markdown("**Query Response**: Please enter a valid question.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    if clear_button:
        st.session_state.query_text = ""
        st.rerun()  # Rerun to update the form with cleared query_text

    st.markdown('</div>', unsafe_allow_html=True)
    if not query_enabled:
        st.markdown("*Please make a prediction and view detailed interpretation first.*")

# Notes
st.markdown("""
    **Notes**: 
    1. This application predicts the ultimate bond strength of FRP-concrete interface using a categorical boosting algorithm optimised with advanced techniques.
    2. The model was trained using data from single-lap shear test experiments.
    3. Uncertainty and reliability metrics provide confidence in predictions, while SHAP and ICE explain feature effects.
""")

# Disclaimer
st.markdown("""
    **Disclaimer**: 
    The explanations provided are based on model outputs and pre-loaded mechanism guidance without external citations.
""")

# Footer
footer = """
<div class="footer">
    <p>© 2025 My Streamlit App. All rights reserved. | Temitope E. Dada, Silas E. Oluwadahunsi, Guobin Gong, Jun Xia, Luigi Di Sarno | For Queries: <a href="mailto: T.Dada19@student.xjtlu.edu.cn"> T.Dada19@student.xjtlu.edu.cn</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)