import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Image Processing App", layout="wide")

st.title("🖼️ Interactive Image Processing App")
st.markdown("Upload an image and apply various transformations to visualize changes in real time.")

# Upload Image
uploaded_file = st.file_uploader("📤 Upload an Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Load image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.subheader("Choose Image Processing Technique")
    option = st.selectbox("Select a method:", (
        "Thresholding",
        "Gaussian Blur",
        "Smoothing",
        "Canny Edge Detection",
        "Contour Detection",
        "Histogram Equalization"
    ))

    processed_image = image.copy()

    if option == "Thresholding":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh_val = st.slider("Threshold Value", 0, 255, 127)
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    elif option == "Gaussian Blur":
        k = st.slider("Kernel Size", 1, 31, 5, step=2)
        processed_image = cv2.GaussianBlur(image, (k, k), 0)

    elif option == "Smoothing":
        kernel = np.ones((9,9),np.float32)/25

        # pass the kernel in the filter2D
        processed_image = cv2.filter2D(image,-1,kernel)

    elif option == "Canny Edge Detection":
        low = st.slider("Low Threshold", 0, 255, 50)
        high = st.slider("High Threshold", 0, 255, 150)
        edges = cv2.Canny(image, low, high)
        processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    elif option == "Contour Detection":
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours
        output = image.copy()
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

        processed_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    elif option == "Histogram Equalization":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)


    # Show Results
    st.markdown("### 🔍 Result Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(processed_image, caption=f"After: {option}", use_container_width=True)
