import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from models.autoencoder import ConvAutoencoder  # Assume this is defined
from utils.video_utils import preprocess_frame, save_anomaly_clip, generate_heatmap  # Assume these are defined
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(page_title="Surveillance Anomaly Using GEN-AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #1e1e2f; color: #ffffff; }
    .stSidebar { background-color: #2a2a3b; }
    .stButton>button { background-color: #4ecca3; color: #ffffff; border-radius: 8px; }
    .stSlider { color: #ffffff; }
    .stMetric { background-color: #2a2a3b; border-radius: 8px; padding: 10px; }
    .stDataFrame { background-color: #2a2a3b; }
    h1, h2, h3 { color: #4ecca3; }
    .alert-box { background-color: #ff4444; padding: 10px; border-radius: 5px; }
    canvas { width: 100% !important; }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = []
if 'frame_buffer' not in st.session_state:
    st.session_state.frame_buffer = []
if 'last_anomaly' not in st.session_state:
    st.session_state.last_anomaly = None
if 'mse_history' not in st.session_state:
    st.session_state.mse_history = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.02
if 'use_video' not in st.session_state:
    st.session_state.use_video = False
if 'uploaded_video' not in st.session_state:
    st.session_state.uploaded_video = None
if 'frame_skip_counter' not in st.session_state:
    st.session_state.frame_skip_counter = 0
if 'caps' not in st.session_state:
    st.session_state.caps = {}
if 'camera_indices' not in st.session_state:
    st.session_state.camera_indices = [0]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists('autoencoder.pth'):
        st.error("Model file 'autoencoder.pth' not found. Please ensure the model is trained and available.")
        logger.error("Model file 'autoencoder.pth' not found.")
        st.stop()
    try:
        model = ConvAutoencoder().to(device)
        model.load_state_dict(torch.load('autoencoder.pth', map_location=device))
        model.eval()
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Ensure clips directory exists
if not os.path.exists('clips/anomalies'):
    os.makedirs('clips/anomalies')
    logger.info("Created clips/anomalies directory.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Live Monitoring", "Analytics", "Project Info"])

# Tab 1: Live Monitoring
with tab1:
    st.header("Live Surveillance Feed")
    col1, col2 = st.columns([3, 1])
    
    FRAME_BUFFER_SIZE = 100
    FRAME_SKIP = 2
    FALLBACK_IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(FALLBACK_IMAGE, "Video Feed Not Available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    with col1:
        video_placeholder = st.empty()
        heatmap_placeholder = st.empty()
    
    with col2:
        st.subheader("Controls")
        input_source = st.radio("Input Source", ["Webcam", "Pre-recorded Video"], help="Choose input source.")
        st.session_state.use_video = (input_source == "Pre-recorded Video")
        
        if st.session_state.use_video and st.session_state.uploaded_video is None:
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
            if uploaded_file is not None:
                video_path = f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.uploaded_video = video_path
                logger.info(f"Uploaded video saved as: {video_path}")
        
        st.multiselect("Select Cameras", options=[i for i in range(4)], default=[0], key="camera_indices")
        start_button = st.button("Start Feed", key="start_feed")
        stop_button = st.button("Stop Feed", key="stop_feed")
        threshold = st.slider("Anomaly Threshold", 0.01, 0.1, st.session_state.threshold, 0.001)
        st.session_state.threshold = threshold
        status_placeholder = st.empty()
        mse_placeholder = st.metric("Reconstruction Error", "0.0000")
        anomaly_count = st.metric("Anomalies Detected", len(st.session_state.anomaly_history))
        clip_count = st.metric("Anomaly Clips Saved", len([f for f in os.listdir('clips/anomalies') if f.endswith('.avi')]))

# Initialize cameras
def init_cameras(indices):
    video_path = st.session_state.uploaded_video or 'test_video.mp4'
    for idx in indices:
        if idx not in st.session_state.caps or not st.session_state.caps[idx].isOpened():
            st.session_state.caps[idx] = cv2.VideoCapture(video_path if st.session_state.use_video else idx)
            if not st.session_state.caps[idx].isOpened():
                logger.error(f"Failed to initialize camera {idx}")
                return False
    return True

# Process a single frame
def process_frame(cap, idx):
    if not cap or not cap.isOpened():
        return None, None, None
    ret, frame = cap.read()
    if not ret:
        if st.session_state.use_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to loop video on camera {idx}")
                return None, None, None
        else:
            logger.error(f"Failed to read frame from camera {idx}")
            return None, None, None
    
    st.session_state.frame_skip_counter += 1
    if st.session_state.frame_skip_counter % FRAME_SKIP != 0:
        return frame, None, None

    try:
        input_tensor = preprocess_frame(frame).to(device)
        input_np = input_tensor.squeeze().cpu().numpy()
        with torch.no_grad():
            output = model(input_tensor)
            mse = torch.mean((output - input_tensor) ** 2).item()
            output_np = output.squeeze().cpu().numpy()
        
        st.session_state.mse_history.append(mse)
        if len(st.session_state.mse_history) > 100:
            st.session_state.mse_history.pop(0)
        
        status = "Anomaly" if mse > st.session_state.threshold else "Normal"
        color = (0, 0, 255) if mse > st.session_state.threshold else (0, 255, 0)
        cv2.putText(frame, f'Status: {status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        st.session_state.frame_buffer.append(frame)
        if len(st.session_state.frame_buffer) > FRAME_BUFFER_SIZE:
            st.session_state.frame_buffer.pop(0)
        
        if mse > st.session_state.threshold and st.session_state.last_anomaly != "Anomaly":
            clip_path = save_anomaly_clip(st.session_state.frame_buffer)
            anomaly_data = {
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "mse": mse,
                "clip_path": clip_path,
                "camera": idx
            }
            st.session_state.anomaly_history.append(anomaly_data)
            st.markdown(f'<div class="alert-box">Anomaly detected on Camera {idx} at {anomaly_data["time"]}. MSE: {mse:.4f}</div>', unsafe_allow_html=True)
            logger.info(f"Anomaly clip saved for Camera {idx}: {clip_path}")
        
        st.session_state.last_anomaly = status
        return frame, mse, output_np
    except Exception as e:
        logger.error(f"Error processing frame for camera {idx}: {str(e)}")
        return None, None, None

# Main processing logic
if start_button:
    st.session_state.running = True
    st.session_state.frame_skip_counter = 0
    if not init_cameras(st.session_state.camera_indices):
        st.error("Failed to initialize cameras.")
        st.session_state.running = False

if stop_button:
    st.session_state.running = False
    for cap in st.session_state.caps.values():
        if cap:
            cap.release()
    st.session_state.caps = {}
    st.session_state.frame_buffer = []
    st.session_state.uploaded_video = None
    logger.info("All video feeds stopped and released.")

if st.session_state.running:
    for idx in st.session_state.camera_indices:
        cap = st.session_state.caps.get(idx)
        frame, mse, output_np = process_frame(cap, idx)
        if frame is None:
            video_placeholder.image(FALLBACK_IMAGE, channels="BGR", caption=f"Camera {idx}", use_container_width=True)
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", caption=f"Camera {idx}", use_container_width=True)
        status_placeholder.markdown(f"**Status (Cam {idx})**: {st.session_state.last_anomaly or 'N/A'}")
        
        if mse is not None:
            mse_placeholder.metric("Reconstruction Error", f"{mse:.4f}")
            if output_np is not None:
                heatmap = generate_heatmap(frame_rgb, output_np, (frame.shape[1], frame.shape[0]))
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap_placeholder.image(heatmap_rgb, channels="RGB", caption=f"Anomaly Heatmap (Cam {idx})", use_container_width=True)
        
        anomaly_count.metric("Anomalies Detected", len(st.session_state.anomaly_history))
        clip_count.metric("Anomaly Clips Saved", len([f for f in os.listdir('clips/anomalies') if f.endswith('.avi')]))

# Tab 2: Analytics
with tab2:
    st.header("Anomaly Analytics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies", len(st.session_state.anomaly_history))
    with col2:
        st.metric("Average MSE", f"{np.mean(st.session_state.mse_history):.4f}" if st.session_state.mse_history else "N/A")
    with col3:
        st.metric("Max MSE", f"{max(st.session_state.mse_history):.4f}" if st.session_state.mse_history else "N/A")

    st.subheader("Reconstruction Error Trend")
    mse_chart_placeholder = st.empty()
    if st.session_state.mse_history:
        mse_data = [{"x": i, "y": mse} for i, mse in enumerate(st.session_state.mse_history)]
        moving_avg = pd.DataFrame({"MSE": st.session_state.mse_history}).rolling(window=5).mean().fillna(0).values.flatten().tolist()
        moving_avg_data = [{"x": i, "y": ma} for i, ma in enumerate(moving_avg)]
        chart_data = {
            "datasets": [
                {"label": "MSE", "data": mse_data, "borderColor": "#4ecca3", "fill": False},
                {"label": "Moving Avg", "data": moving_avg_data, "borderColor": "#ffffff", "fill": False}
            ]
        }
        st.markdown("""
        <canvas id="mseChart" style="max-height: 300px;"></canvas>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        const ctx = document.getElementById('mseChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: """ + json.dumps(chart_data) + """,
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'MSE Trend with Moving Average', color: '#ffffff' } },
                scales: {
                    x: { title: { display: true, text: 'Time', color: '#ffffff' } },
                    y: { title: { display: true, text: 'MSE', color: '#ffffff' } }
                }
            }
        });
        </script>
        """, unsafe_allow_html=True)

    st.subheader("Anomaly Distribution")
    hist_placeholder = st.empty()
    if st.session_state.anomaly_history:
        mse_values = [entry["mse"] for entry in st.session_state.anomaly_history]
        bins = np.histogram_bin_edges(mse_values, bins=20)
        hist_data = np.histogram(mse_values, bins=bins)[0].tolist()
        hist_chart_data = {
            "labels": [f"{b:.2f}" for b in bins[:-1]],
            "datasets": [{"label": "Anomaly Count", "data": hist_data, "backgroundColor": "#4ecca3"}]
        }
        st.markdown("""
        <canvas id="histChart" style="max-height: 300px;"></canvas>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        const histCtx = document.getElementById('histChart').getContext('2d');
        new Chart(histCtx, {
            type: 'bar',
            data: """ + json.dumps(hist_chart_data) + """,
            options: {
                responsive: true,
                plugins: { title: { display: true, text: 'Anomaly Distribution by MSE', color: '#ffffff' } },
                scales: {
                    x: { title: { display: true, text: 'MSE', color: '#ffffff' } },
                    y: { title: { display: true, text: 'Count', color: '#ffffff' } }
                }
            }
        });
        </script>
        """, unsafe_allow_html=True)

    st.subheader("Anomaly History")
    history_placeholder = st.empty()
    if st.session_state.anomaly_history:
        history_data = [{"Time": h["time"], "MSE": f"{h['mse']:.4f}", "Clip": h["clip_path"], "Camera": h.get("camera", 0)} for h in st.session_state.anomaly_history]
        history_placeholder.dataframe(history_data)
        if st.button("Export to CSV", key="export_csv"):
            df = pd.DataFrame(history_data)
            df.to_csv("anomaly_history.csv", index=False)
            st.success("Exported to anomaly_history.csv")

    st.subheader("Saved Anomaly Clips")
    clip_files = [f for f in os.listdir('clips/anomalies') if f.endswith('.avi')]
    clip_select = st.selectbox("Select a Clip to View", clip_files if clip_files else ["No clips available"])
    if clip_select and clip_select != "No clips available":
        clip_path = os.path.join('clips/anomalies', clip_select)
        st.video(clip_path)

# Tab 3: Project Info
with tab3:
    st.header("About Smart Surveillance")
    st.markdown("""
    **Smart Surveillance** is a generative AI-powered system for real-time anomaly detection, built for hackathon demos.

    ### Technical Details
    - **Model**: Convolutional Autoencoder.
    - **Training**: UCSD Ped1 dataset.
    - **Inference**: Real-time detection, heatmap visualization.
    - **Features**: Multi-camera, data export, anomaly clip saving.

    ### Use Cases in India
    - **Public Safety**: Railway stations.
    - **Traffic Management**: Highways.
    - **Retail Security**: Malls.
    - **Industrial Safety**: Industrial corridors.
    - **Rural Surveillance**: Farms.

    ### Future Enhancements
    - Cloud storage.
    - Real-time SMS alerts.
    - AI model retraining.
    """)

# Cleanup on script end
if not st.session_state.running:
    for cap in st.session_state.caps.values():
        if cap:
            cap.release()
    st.session_state.caps = {}
    logger.info("All video feeds released at script end.")