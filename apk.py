import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants
WIDTH, HEIGHT = 800, 600
DEMO_IMAGE = 'stand.jpg'
MODEL_FILE = 'graph_opt.pb'

# Body parts and pose pairs for visualization
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        net = cv2.dnn.readNetFromTensorflow(MODEL_FILE)
        return net
    except Exception as e:
        st.error("Error loading the model. Make sure 'graph_opt.pb' is in the directory.")
        st.stop()

# Pose estimation function

def pose_detector(frame, net, threshold):
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # Prepare the frame for the model
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Focus on the first 19 body parts

    points = []
    for i in range(len(BODY_PARTS)):
        heatmap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x = int((frame_width * point[0]) / out.shape[3])
        y = int((frame_height * point[1]) / out.shape[2])
        points.append((x, y) if conf > threshold else None)

    # Draw key points and connections
    for pair in POSE_PAIRS:
        part_from, part_to = pair
        id_from = BODY_PARTS[part_from]
        id_to = BODY_PARTS[part_to]

        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
            cv2.ellipse(frame, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Streamlit app
st.title("Human Pose Estimation using OpenCV")
st.markdown("Upload an image to detect human poses. Ensure the image is clear with all body parts visible.")

# Upload image
img_file_buffer = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    image = np.array(Image.open(DEMO_IMAGE))

# Display original image
st.subheader("Original Image")
st.image(image, caption="Original Image", use_column_width=True)

# Threshold slider
threshold = st.slider("Threshold for detecting key points", min_value=0, max_value=100, value=20, step=5) / 100

# Load the model
net = load_model()

# Perform pose detection
output_image = pose_detector(image, net, threshold)

# Display output image
st.subheader("Pose Estimated Image")
st.image(output_image, caption="Pose Estimated Image", use_column_width=True)

# Download the output image
result_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
is_success, buffer = cv2.imencode(".jpg", result_image)
st.download_button(label="Download Pose Detected Image", data=buffer.tobytes(), file_name="pose_detected.jpg", mime="image/jpeg")

st.markdown("---")

