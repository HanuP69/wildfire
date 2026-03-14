import os
import torch
import xgboost as xgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Evidently AI imports
from evidently import Report
from evidently.presets import DataDriftPreset

from src.model.wildfire import CNNLSTMModel, WildfireFusionModel
from src.data.firms_client import generate_mock_tensor

app = FastAPI(title="NASA FIRMS Wildfire Prediction API", version="1.0.0")

# Directories and Model Paths
MODEL_DIR = os.getenv("MODEL_DIR", "models")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Globals for models
cnn_lstm_model = None
fusion_model = None
xgb_model = None
meta_learner = None

# For Evidently AI Data Drift
reference_data = None
current_data_log = []

class PredictRequest(BaseModel):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

def extract_tabular_from_tensor(patch: np.ndarray) -> np.ndarray:
    """Extracts mean, std, max, min, median per channel into a (60,) array."""
    # patch is (64, 64, 12). Reshape to (4096, 12)
    flat = patch.reshape(-1, 12)
    feats = [
        np.mean(flat, axis=0),
        np.std(flat, axis=0),
        np.max(flat, axis=0),
        np.min(flat, axis=0),
        np.median(flat, axis=0)
    ]
    return np.concatenate(feats).astype(np.float32)

@app.on_event("startup")
def load_models():
    global cnn_lstm_model, fusion_model, xgb_model, meta_learner, reference_data
    
    print("Loading models...")
    
    # 1. Load PyTorch Models
    try:
        cnn_lstm_path = os.path.join(MODEL_DIR, "CNN_LSTM_best.pt")
        if os.path.exists(cnn_lstm_path):
            cnn_lstm_model = CNNLSTMModel().to(DEVICE)
            cnn_lstm_model.load_state_dict(torch.load(cnn_lstm_path, map_location=DEVICE))
            cnn_lstm_model.eval()
            print("Loaded CNN+LSTM model.")
        else:
            print(f"Warning: {cnn_lstm_path} not found. Mocking model.")
    except Exception as e:
        print(f"Failed to load CNN+LSTM: {e}")

    try:
        fusion_path = os.path.join(MODEL_DIR, "FullFusion_best.pt")
        if os.path.exists(fusion_path):
            fusion_model = WildfireFusionModel().to(DEVICE)
            fusion_model.load_state_dict(torch.load(fusion_path, map_location=DEVICE))
            fusion_model.eval()
            print("Loaded Full Fusion model.")
        else:
            print(f"Warning: {fusion_path} not found. Mocking model.")
    except Exception as e:
        print(f"Failed to load Full Fusion: {e}")

    # 2. Load XGBoost
    try:
        xgb_path = os.path.join(MODEL_DIR, "xgboost_model.json")
        if os.path.exists(xgb_path):
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(xgb_path)
            print("Loaded XGBoost model.")
        else:
            print(f"Warning: {xgb_path} not found. Mocking model.")
    except Exception as e:
        print(f"Failed to load XGBoost: {e}")

    # 3. Load Meta Learner (Logistic Regression)
    try:
        meta_path = os.path.join(MODEL_DIR, "meta_learner.pkl")
        if os.path.exists(meta_path):
            meta_learner = joblib.load(meta_path)
            print("Loaded Meta Learner.")
        else:
            print(f"Warning: {meta_path} not found. Mocking model.")
    except Exception as e:
        print(f"Failed to load Meta Learner: {e}")
        
    # Generate mock reference data for Evidently Drift Report
    # Realistically, you'd load X_train tabular stats here
    reference_data = pd.DataFrame(np.random.randn(100, 60))
    print("Models loaded successfully.")

@app.post("/predict")
def predict(req: PredictRequest):
    """
    1. Fetch FIRMS Data
    2. Format inputs (img, seq, tabular)
    3. Run XGBoost, CNN+LSTM, Full Fusion
    4. Run Meta Learner Stack
    5. Log for Evidently Drift
    """
    try:
        # 1. Fetch live NASA data & mock background
        patch = generate_mock_tensor(req.min_lon, req.min_lat, req.max_lon, req.max_lat)
        
        # 2. Extract inputs
        # CNN img: (1, 12, 64, 64)
        img = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        # LSTM seq: (1, 64, 12)
        seq = torch.from_numpy(patch).mean(dim=1).unsqueeze(0).to(DEVICE)
        
        # Tabular: (1, 60)
        tab_np = extract_tabular_from_tensor(patch)
        tab = torch.from_numpy(tab_np).unsqueeze(0).to(DEVICE)
        tab_2d = tab_np.reshape(1, -1)
        
        # Log tabular for drift monitoring
        current_data_log.append(tab_np)
        
        # Initialize probabilities
        cnn_lstm_prob = 0.5
        fusion_prob = 0.5
        xgb_prob = 0.5
        meta_prob = 0.5
        
        # 3. Run Inference
        with torch.no_grad():
            if cnn_lstm_model:
                logits = cnn_lstm_model(img, seq, tab)
                cnn_lstm_prob = torch.sigmoid(logits).item()
                
            if fusion_model:
                logits = fusion_model(img, seq, tab)
                fusion_prob = torch.sigmoid(logits).item()
                
        if xgb_model:
            xgb_prob = float(xgb_model.predict_proba(tab_2d)[0, 1])
            
        # 4. Meta Learner
        if meta_learner:
            stack_input = np.array([[xgb_prob, cnn_lstm_prob, fusion_prob]])
            meta_prob = float(meta_learner.predict_proba(stack_input)[0, 1])
        else:
            # Fallback ensemble average if meta model is missing
            meta_prob = (xgb_prob + cnn_lstm_prob + fusion_prob) / 3.0
            
        return {
            "bbox": req.dict(),
            "probabilities": {
                "xgboost": round(xgb_prob, 4),
                "cnn_lstm": round(cnn_lstm_prob, 4),
                "full_fusion": round(fusion_prob, 4),
                "stacked_meta": round(meta_prob, 4)
            },
            "status": "success",
            "fire_points_detected": int(np.sum(patch[:, :, 11]))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drift-report")
def get_drift_report():
    """Generates an Evidently Data Drift Report using logged incoming requests."""
    if len(current_data_log) < 5:
        return {"message": "Not enough data collected for drift report. Minimum 5 requests required."}
        
    current_df = pd.DataFrame(current_data_log)
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_df)
    
    report_dict = report.as_dict()
    dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
    
    return {
        "drift_detected": dataset_drift,
        "total_requests_logged": len(current_data_log),
        "timestamp": pd.Timestamp.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
