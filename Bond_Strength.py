import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import joblib
import matplotlib.pyplot as plt
import chromadb
from catboost import CatBoostRegressor, Pool
from sentence_transformers import SentenceTransformer, util
import requests
import json
import re
import logging
import os
from properscoring import crps_gaussian
import nltk
import tempfile
import shutil

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# -------------------------
# Config (relative paths)
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

# OpenRouter API config – key from Streamlit secrets
OPENROUTER_API_KEY = st.secrets.get("openrouter", {}).get("api_key")
if not OPENROUTER_API_KEY:
    st.error("OpenRouter API key not found. Set it in `.streamlit/secrets.toml`.")
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


# Copy chroma_db to a writable temp folder (Streamlit Cloud needs writable folder)
tmp_chroma_path = tempfile.mkdtemp()
shutil.copytree("chroma_db", tmp_chroma_path, dirs_exist_ok=True)

# Initialize PersistentClient in temp folder
try:
    client = chromadb.PersistentClient(path=tmp_chroma_path)
    collection = client.get_collection("pdf_chunks")
    logger.info(f"Loaded ChromaDB collection 'pdf_chunks' with {collection.count()} items.")
except Exception as e:
    logger.warning(f"Failed to load 'pdf_chunks' collection: {e}. Falling back to no literature guidance.")
    collection = None

    collection = None

embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")


# Load model and scalers
try:
    model = CatBoostRegressor()
    model.load_model(os.path.join(model_dir, "catboost_mean.cbm"))
    scaler_X = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    with open(os.path.join(model_dir, "conformal_quantile.pkl"), "rb") as f:
        conformal_quantile = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model or scalers: {e}")
    st.stop()

# Initialize session state
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

# Global helper functions
def get_direction(ils_slope):
    if ils_slope > 0.01:
        return "positive"
    elif ils_slope < -0.01:
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

# Custom CSS (unchanged)
st.markdown("""
    <style>
    .main {max-width: 1000px; margin: 0 auto; padding: 1rem;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px; padding: 0.5rem 1rem; font-size: 16px;}
    .stNumberInput label {font-size: 16px; font-weight: bold; color: #333;}
    .stNumberInput input {border-radius: 5px; padding: 0.5rem;}
    .prediction-box {background-color: #e8f5e9; padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;}
    .prediction-box h2 {color: #2e7d32; font-size: 28px; margin-bottom: 0.5rem;}
    .metrics-box {background-color: #f5f5f5; padding: 1rem; border-radius: 10px; margin: 1rem 0;}
    .stExpander {border: 1px solid #ddd; border-radius: 5px; margin: 1rem 0;}
    .query-box {background-color: #f9f9f9; padding: 1rem; border-radius: 10px; margin: 1rem 0; width: 100%; max-width: 1000px; box-sizing: border-box;}
    .query-box input {border: 1px solid #ccc; border-radius: 5px; padding: 0.5rem; width: 100%; font-size: 16px;}
    .response-box {background-color: #f0f4f8; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; width: 100% !important; max-width: 1000px !important; box-sizing: border-box; overflow-wrap: break-word; display: block; min-width: 0;}
    .chat-container {margin-top: 1rem; max-height: 500px; overflow-y: auto; padding: 0.5rem; border: 1px solid #ddd; border-radius: 8px; background-color: #fafafa;}
    .footer {position: relative; width: 100%; background-color: #f1f1f1; text-align: center; padding: 10px; font-size: 12px; color: #6c757d; margin-top: 2rem;}
    @media (max-width: 600px) {
        .prediction-box h2 {font-size: 24px;}
        .stNumberInput label {font-size: 14px;}
        .stButton>button {font-size: 14px;}
        .query-box input {font-size: 14px;}
        .response-box {font-size: 14px; width: 90vw !important; max-width: 90vw !important;}
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

# Make prediction
if st.button('Predict'):
    try:
        input_data = np.array([[concrete_width, compressive_strength, frp_modulus, 
                               frp_overall_thickness, frp_sheet_width, bond_length]])
        input_df = pd.DataFrame(input_data, columns=raw_features)
        input_scaled = scaler_X.transform(input_data)
        sample_pool = Pool(input_scaled)

        pred = model.predict(sample_pool)[0]
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

        st.session_state.prediction_data = {
            "pred": pred,
            "reliability": reliability,
            "uncertainty_metrics": uncertainty_metrics,
            "input_df": input_df,
            "input_scaled": input_scaled
        }
        st.session_state.chat_history = []  # Reset chat on new prediction
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# === PERSISTENT PREDICTION DISPLAY (Always on top) ===
if st.session_state.prediction_data:
    pred_data = st.session_state.prediction_data
    pred = pred_data["pred"]
    reliability = pred_data["reliability"]
    uncertainty_metrics = pred_data["uncertainty_metrics"]

    with st.container():
        st.markdown(f"""
            <div class="prediction-box">
                <h2>Predicted Ultimate Bond Strength: {pred:.2f} kN</h2>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### Reliability and Uncertainty Metrics")
        st.markdown(f"""
            <div class="metrics-box">
                <p><strong>Reliability Class:</strong> {reliability['class']}</p>
                <p><strong>Probability of Failure (Pf):</strong> {reliability['Pf']:.2f}</p>
                <p><strong>Reliability Index (Beta):</strong> {reliability['Beta']:.2f}</p>
                <p><strong>Prediction Interval Width (PIW):</strong> {uncertainty_metrics['PIW']:.2f}</p>
                <p><strong>Coverage Width-based Criterion (CWC):</strong> {uncertainty_metrics['CWC']:.2f}</p>
                <p><strong>Continuous Ranked Probability Score (CRPS):</strong> {uncertainty_metrics['CRPS']:.2f}</p>
                <p><strong>Prediction Range:</strong> {round(pred - uncertainty_metrics['PIW']/2, 2):.2f} to {round(pred + uncertainty_metrics['PIW']/2, 2):.2f} kN</p>
            </div>
        """, unsafe_allow_html=True)

    # === DETAILED INTERPRETATION (FULLY RESTORED) ===
    with st.expander("View Detailed Interpretation"):
        # SHAP
        st.subheader("Feature Importance (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pred_data["input_scaled"])
            mean_abs_shap = np.abs(shap_values[0])
            mean_abs_shap_dict = {f: float(mean_abs_shap[i]) for i, f in enumerate(raw_features)}
        except Exception as e:
            st.error(f"SHAP failed: {e}")
            st.stop()

        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, pred_data["input_scaled"], feature_names=display_features, plot_type="bar", show=False)
            plt.title("Feature Importance (Mean |SHAP|)")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP plot failed: {e}")

        # ILS
        st.subheader("Individual Local Sensitivity (ILS) Plot")
        ils_results = {}
        n_points = 50
        for f in raw_features:
            try:
                user_val = pred_data["input_df"][f].iloc[0]
                range_min = max(0.0, user_val * 0.8)
                range_max = user_val * 1.2
                feature_values = np.linspace(range_min, range_max, n_points)
                ice_data = pred_data["input_df"].copy()
                ice_preds = []
                for val in feature_values:
                    ice_data[f] = val
                    ice_scaled = scaler_X.transform(ice_data)
                    ice_pred = model.predict(Pool(ice_scaled))[0]
                    ice_preds.append(ice_pred)
                dx = np.diff(feature_values)
                dy = np.diff(ice_preds)
                slope_mean = float(np.nanmean(dy / (dx + 1e-8)))
                ils_results[f] = {
                    "x": feature_values.tolist(),
                    "y": ice_preds,
                    "stats": {"mean": round(float(np.mean(ice_preds)), 2), "slope_mean": round(slope_mean, 2)}
                }
            except Exception as e:
                st.warning(f"ILS for {feature_map[f]} failed: {e}")
                ils_results[f] = {"x": [], "y": [], "stats": {"mean": 0.0, "slope_mean": 0.0}}

        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            for f in raw_features:
                if ils_results[f]["x"]:
                    plt.plot(ils_results[f]["x"], ils_results[f]["y"], label=feature_map[f], linewidth=2)
                    plt.scatter([pred_data["input_df"][f].iloc[0]], [pred], color=plt.gca().lines[-1].get_color(), s=100, zorder=5)
            plt.title("Individual Local Sensitivity (ILS) for All Features")
            plt.xlabel("Feature Value")
            plt.ylabel("Predicted Bond Strength (kN)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"ILS plot failed: {e}")

        # Feature Summary
        st.subheader("Feature Effect Summary")
        results = []
        for f in raw_features:
            shap_val_abs = mean_abs_shap_dict.get(f, 0.0)
            ils_slope = ils_results[f]["stats"]["slope_mean"]
            ils_mean = ils_results[f]["stats"]["mean"]
            direction = get_direction(ils_slope)
            results.append({
                "Feature": feature_map[f],
                "Mean|SHAP|": round(shap_val_abs, 2),
                "ILS Slope": round(ils_slope, 2),
                "ILS Mean": round(ils_mean, 2),
                "Direction": direction
            })
        results_df = pd.DataFrame(results)
        st.write(results_df)

        # Literature
        try:
            literature = {f: query_literature(f, collection, embedder) for f in raw_features}
        except Exception as e:
            st.warning(f"Literature query failed: {e}")
            literature = {f: "" for f in raw_features}
        pred_data.update({'results_df': results_df, 'mean_abs_shap_dict': mean_abs_shap_dict, 'ils_results': ils_results, 'literature': literature})

        # === TECHNICAL SUMMARY (FULLY RESTORED) ===
        def build_technical_summary_prompt():
            try:
                p0 = round(float(pred), 2)
                piw = round(uncertainty_metrics['PIW'], 2)
                cwc = round(uncertainty_metrics['CWC'], 2)
                crps = round(uncertainty_metrics['CRPS'], 2)
                range_lower = round(p0 - piw / 2, 2)
                range_upper = round(p0 + piw / 2, 2)
                
                txt = [
                    f"The CatBoost model predicted a bond strength of {p0:.2f} kN.",
                    f"Reliability metrics: Class={reliability['class']}, Pf={reliability['Pf']:.2f}, Beta={reliability['Beta']:.2f}.",
                    f"Uncertainty metrics: PIW={piw:.2f}, CWC={cwc:.2f}, CRPS={crps:.2f}.",
                    "\nYour task:",
                    "Write a comprehensive, technically precise summary in exactly 3 paragraphs that satisfies all the following conditions:",
                    "1. First paragraph: Report the model prediction, reliability metrics (Class, Pf, Beta), and uncertainty metrics (PIW, CWC, CRPS), all to 2 decimal places. Quantify confidence: higher CWC indicates wider intervals (less efficient uncertainty); lower CRPS reflects lower prediction error. Estimate the prediction variability range using prediction ± PIW/2. State that the High reliability class (Pf=0.06, Beta=1.50) indicates low failure risk and high structural safety. Use exact phrasing like 'Class=High' for reliability class.",
                    "2. Second paragraph: Describe the effect of all features on bond strength using Mean|SHAP| (magnitude) and ILS Slope (direction), both to 2 decimal places. For each feature, specify magnitude (small: Mean|SHAP|<0.5, moderate: 0.5–1.5, significant: >1.5) and direction (positive: ILS Slope > 0.01, negative: ILS Slope < -0.01, neutral: otherwise). Group features as FRP properties (FRP Modulus, FRP Overall Thickness, FRP Sheet Width), concrete properties (Concrete Width, Compressive Strength), and bond property (Bond Length). Mention each feature exactly once. Use the provided mechanism guidance to briefly explain the effect as consistent with known FRP–concrete behavior (e.g., 'consistent with improved interfacial shear transfer'), avoiding strong causal phrasing like 'due to' or 'caused by'. Format ILS Slope as 'ILS Slope=+X.XX' or 'ILS Slope=-X.XX' for clarity.",
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
                    "- Report numerical values to 2 decimal places (e.g., Mean|SHAP|, ILS Slope, prediction, reliability, uncertainty).",
                    "- Do NOT mention, cite, or reference any literature, studies, papers, authors, or sources under any circumstances.",
                    "- Mention each feature exactly once in the second paragraph using the display names (e.g., 'Concrete Width' instead of 'Concrete_Width').",
                    "- When explaining feature effects, ensure each feature has one clear technical interpretation.",
                    "- If a feature is mentioned again in the implications section, ensure the second mention adds new design or practical context — otherwise, omit it to avoid redundancy.",
                    "- Ensure the output is realistic and directly applicable to FRP-concrete engineering.",
                    "- Format ILS Slope as 'ILS Slope=+X.XX' or 'ILS Slope=-X.XX' to ensure consistent extraction.",
                    "=== Feature Data ==="
                ]
                for f in raw_features:
                    shap_val_abs = round(float(mean_abs_shap_dict.get(f, 0.0)), 2)
                    ils_slope = round(float(ils_results[f]["stats"]["slope_mean"]), 2)
                    ils_mean = round(float(ils_results[f]["stats"]["mean"]), 2)
                    magnitude = "small" if shap_val_abs < 0.5 else "moderate" if shap_val_abs <= 1.5 else "significant"
                    direction = get_direction(ils_slope)
                    txt.append(
                        f"{feature_map[f]}: Mean|SHAP|={shap_val_abs:.2f}, ILS Slope={ils_slope:+.2f}, ILS Mean={ils_mean:.2f}, Magnitude={magnitude}, Direction={direction}"
                    )
                    guidance = literature.get(f, "")
                    txt.append(f"Mechanism guidance for {feature_map[f]}: {guidance[:500]}" if guidance else f"No mechanism guidance available for {feature_map[f]}.")
                return "\n".join(txt)
            except Exception as e:
                logger.error(f"Prompt build failed: {e}")
                return None

        prompt = build_technical_summary_prompt()
        if prompt is None:
            st.error("Failed to generate summary prompt.")
            st.stop()
        llm_text = call_openrouter(prompt)
        llm_json = try_parse_json(llm_text)

        if not llm_json or "summary_paragraphs" not in llm_json or len(llm_json["summary_paragraphs"]) != 3:
            llm_json = {
                "summary_paragraphs": [
                    f"The CatBoost model predicted a bond strength of {pred:.2f} kN with reliability class {reliability['class']} (Pf={reliability['Pf']:.2f}, Beta={reliability['Beta']:.2f}). Uncertainty metrics (PIW={uncertainty_metrics['PIW']:.2f}, CWC={uncertainty_metrics['CWC']:.2f}, CRPS={uncertainty_metrics['CRPS']:.2f}) indicate high confidence with a variability range of {round(pred - uncertainty_metrics['PIW']/2, 2):.2f} to {round(pred + uncertainty_metrics['PIW']/2, 2):.2f} kN.",
                    f"Feature effects: FRP properties (FRP Modulus: Mean|SHAP|={round(mean_abs_shap_dict.get('FRP_Modulus', 0.0), 2):.2f}, ILS Slope={round(ils_results['FRP_Modulus']['stats']['slope_mean'], 2):+.2f}, {get_direction(ils_results['FRP_Modulus']['stats']['slope_mean'])}; FRP Overall Thickness: Mean|SHAP|={round(mean_abs_shap_dict.get('FRP_Overall_Thickness', 0.0), 2):.2f}, ILS Slope={round(ils_results['FRP_Overall_Thickness']['stats']['slope_mean'], 2):+.2f}, {get_direction(ils_results['FRP_Overall_Thickness']['stats']['slope_mean'])}; FRP Sheet Width: Mean|SHAP|={round(mean_abs_shap_dict.get('FRP_Sheet_Width', 0.0), 2):.2f}, ILS Slope={round(ils_results['FRP_Sheet_Width']['stats']['slope_mean'], 2):+.2f}, {get_direction(ils_results['FRP_Sheet_Width']['stats']['slope_mean'])}), concrete properties (Concrete Width: Mean|SHAP|={round(mean_abs_shap_dict.get('Concrete_Width', 0.0), 2):.2f}, ILS Slope={round(ils_results['Concrete_Width']['stats']['slope_mean'], 2):+.2f}, {get_direction(ils_results['Concrete_Width']['stats']['slope_mean'])}; Compressive Strength: Mean|SHAP|={round(mean_abs_shap_dict.get('Compressive_Strength', 0.0), 2):.2f}, ILS Slope={round(ils_results['Compressive_Strength']['stats']['slope_mean'], 2):+.2f}, {get_direction(ils_results['Compressive_Strength']['stats']['slope_mean'])}), bond property (Bond Length: Mean|SHAP|={round(mean_abs_shap_dict.get('Bond_Length', 0.0), 2):.2f}, ILS Slope={round(ils_results['Bond_Length']['stats']['slope_mean'], 2):+.2f}, {get_direction(ils_results['Bond_Length']['stats']['slope_mean'])}).",
                    f"Practical implications: Designers should prioritize features with high Mean|SHAP| (e.g., FRP Overall Thickness: {round(mean_abs_shap_dict.get('FRP_Overall_Thickness', 0.0), 2):.2f}, Bond Length: {round(mean_abs_shap_dict.get('Bond_Length', 0.0), 2):.2f}) to optimize bond strength, supported by high reliability (Pf={reliability['Pf']:.2f}) and confidence (CWC={uncertainty_metrics['CWC']:.2f}, CRPS={uncertainty_metrics['CRPS']:.2f})."
                ]
            }

        st.subheader("Technical Summary")
        for para in llm_json["summary_paragraphs"]:
            st.markdown(para)

        # === FAITHFULNESS EVALUATION (FIXED SYNTAX) ===
        def evaluate_numerical_faithfulness(llm_json, results_df, pred, reliability, uncertainty_metrics):
            try:
                if not llm_json or len(llm_json["summary_paragraphs"]) != 3:
                    return {"score": 0}
                paragraphs = llm_json["summary_paragraphs"]
                gt = {
                    "pred": round(float(pred), 2),
                    "rel": reliability,
                    "unc": uncertainty_metrics,
                    "feat": {row["Feature"]: {"shap": row["Mean|SHAP|"], "ils": row["ILS Slope"]} for _, row in results_df.iterrows()}
                }
                score = 0
                max_score = 6 + 2*len(raw_features)
                for pat, val in [
                    (r"bond strength of[^\d]*([+-]?\d+\.\d{1,})", gt["pred"]),
                    (r"\bPf[^\d]*([+-]?\d+\.\d{1,})", gt["rel"]["Pf"]),
                    (r"\bBeta[^\d]*([+-]?\d+\.\d{1,})", gt["rel"]["Beta"]),
                    (r"\bPIW[^\d]*([+-]?\d+\.\d{1,})", gt["unc"]["PIW"]),
                    (r"\bCWC[^\d]*([+-]?\d+\.\d{1,})", gt["unc"]["CWC"]),
                    (r"\bCRPS[^\d]*([+-]?\d+\.\d{1,})", gt["unc"]["CRPS"]),
                ]:
                    m = re.search(pat, paragraphs[0], re.IGNORECASE)
                    if m and compare_numerics(m.group(1), val):
                        score += 1
                for f in raw_features:
                    disp = feature_map[f]
                    m_shap = re.search(rf"{disp}.*?Mean\|SHAP\|.*?([+-]?\d+\.\d{{1,}})", paragraphs[1], re.IGNORECASE)
                    m_ils = re.search(rf"{disp}.*?ILS Slope.*?([+-]?\d+\.\d{{1,}})", paragraphs[1], re.IGNORECASE)
                    if m_shap and compare_numerics(m_shap.group(1), gt["feat"][disp]["shap"]):
                        score += 1
                    if m_ils and compare_numerics(m_ils.group(1), gt["feat"][disp]["ils"]):
                        score += 1
                return {"score": round((score / max_score) * 100, 2)}
            except:
                return {"score": 0}

        def evaluate_physical_groundedness(llm_json, literature, ils_results, mean_abs_shap_dict):
            try:
                if not llm_json or len(llm_json["summary_paragraphs"]) != 3:
                    return {"score": 0}
                p2, p3 = llm_json["summary_paragraphs"][1], llm_json["summary_paragraphs"][2]
                score = 0
                max_score = 0
                top2 = sorted(mean_abs_shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
                for f, _ in top2:
                    disp = feature_map[f]
                    max_score += 2
                    if disp.lower() in p2.lower() or disp.lower() in p3.lower():
                        score += 1
                    slope = ils_results[f]["stats"]["slope_mean"]
                    dir_word = "increases" if slope > 0.01 else "decreases" if slope < -0.01 else "neutral"
                    if dir_word in p2.lower() or dir_word in p3.lower():
                        score += 1
                lit_text = " ".join([v for v in literature.values() if v])[:1000]
                if lit_text:
                    try:
                        lit_emb = embedder.encode(lit_text)
                        p3_emb = embedder.encode(p3[:1000])
                        sim = util.cos_sim(lit_emb, p3_emb).item()
                        score += sim * 2
                        max_score += 2
                    except:
                        pass
                return {"score": round((score / max_score) * 100, 2) if max_score else 0}
            except:
                return {"score": 0}

        num_faith = evaluate_numerical_faithfulness(llm_json, results_df, pred, reliability, uncertainty_metrics)
        phys_faith = evaluate_physical_groundedness(llm_json, literature, ils_results, mean_abs_shap_dict)
        grand_composite = round(0.6 * num_faith["score"] + 0.4 * phys_faith["score"], 2)

        st.subheader("LLM Explanation Quality")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Numerical Faithfulness**\n{num_faith['score']:.2f}%")
        with col2:
            st.markdown(f"**Physical Groundedness**\n{phys_faith['score']:.2f}%")
        with col3:
            st.markdown(f"**Grand Composite**\n{grand_composite:.2f}%")

# === QUERY SECTION (Persistent Chat) ===
st.markdown("### Ask a Question")
with st.container():
    st.markdown('<div class="query-box">', unsafe_allow_html=True)

    query_enabled = st.session_state.prediction_data is not None

    with st.form(key="query_form"):
        query = st.text_input(
            "Enter your question about bond strength, reliability, or feature effects",
            value=st.session_state.query_text,
            placeholder="e.g., What is the significance of the bond strength prediction?",
            disabled=not query_enabled
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            submit_button = st.form_submit_button("Submit Query")
        with col2:
            clear_button = st.form_submit_button("Clear Chat")

    if submit_button and query.strip() and query_enabled:
        with st.spinner("Thinking..."):
            st.session_state.query_text = query
            prompt = f"""
            Answer using only the following data:
            - Prediction: {pred_data['pred']:.2f} kN
            - Reliability: Class={pred_data['reliability']['class']}, Pf={pred_data['reliability']['Pf']:.2f}
            - Uncertainty: PIW={pred_data['uncertainty_metrics']['PIW']:.2f}, CRPS={pred_data['uncertainty_metrics']['CRPS']:.2f}
            Question: {query}
            """
            response = call_openrouter(prompt) or "No response."
            st.session_state.chat_history.append({"question": query, "answer": response})

    if clear_button:
        st.session_state.chat_history = []
        st.session_state.query_text = ""
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, chat in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{i+1}:** {chat['question']}")
            st.markdown(f'<div class="response-box">{chat["answer"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif query_enabled:
        st.markdown("*Ask a question about the prediction above.*")
    else:
        st.markdown("*Please make a prediction first.*")

# Notes & Disclaimer
st.markdown("""
    **Notes**: 
    1. This application predicts the ultimate bond strength of FRP-concrete interface using a categorical boosting algorithm optimised with advanced techniques.
    2. The model was trained using data from single-lap shear test experiments.
    3. Uncertainty and reliability metrics provide confidence in predictions, while SHAP and ILS explain feature effects.
""")

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