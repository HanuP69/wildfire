import os
import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

def get_prediction(min_lon, min_lat, max_lon, max_lat):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Format the probabilities as Markdown
        probs = data.get("probabilities", {})
        md_probs = f"""
### Model Probabilities
- **XGBoost:** {probs.get('xgboost', 0):.2%}
- **CNN+LSTM:** {probs.get('cnn_lstm', 0):.2%}
- **Full Fusion:** {probs.get('full_fusion', 0):.2%}
- **Stacked Meta-Learner:** {probs.get('stacked_meta', 0):.2%}
        """
        
        status = f"✅ Success. Fetched FIRMS active fires: {data.get('fire_points_detected', 0)} points."
        return md_probs, status
        
    except requests.exceptions.RequestException as e:
        return "⚠️ Error connecting to API.", str(e)

def get_drift():
    try:
        response = requests.get(f"{API_URL}/drift-report", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "message" in data:
            return data["message"]
            
        drift = data.get("drift_detected", False)
        status = "🔴 Drift Detected!" if drift else "🟢 No Drift Detected"
        return f"{status} (Total reqs: {data.get('total_requests_logged', 0)})"
    except Exception as e:
        return f"⚠️ Could not fetch drift report: {e}"

# Build Gradio UI
with gr.Blocks(title="Wildfire Spread Predictor", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🔥 Wildfire Spread Prediction — Live Dashboard")
    gr.Markdown("Uses NASA FIRMS active fire data combined with expected terrain/weather to predict NEXT DAY fire spread using a Stacked Neural Ensemble.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Select Region (Bounding Box)")
            min_lon = gr.Number(label="Min Longitude (West)", value=-120.0)
            min_lat = gr.Number(label="Min Latitude (South)", value=34.0)
            max_lon = gr.Number(label="Max Longitude (East)", value=-119.0)
            max_lat = gr.Number(label="Max Latitude (North)", value=35.0)
            
            predict_btn = gr.Button("Fetch NASA Data & Predict", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Prediction Results")
            results_md = gr.Markdown("No predictions yet.")
            status_text = gr.Textbox(label="Status", interactive=False)
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 3. Evidently AI Data Drift Monitor")
            drift_out = gr.Textbox(label="Drift Status", interactive=False)
            drift_btn = gr.Button("Check Data Drift")
            

    # Actions
    predict_btn.click(
        fn=get_prediction,
        inputs=[min_lon, min_lat, max_lon, max_lat],
        outputs=[results_md, status_text]
    )
    
    drift_btn.click(
        fn=get_drift,
        inputs=[],
        outputs=[drift_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
