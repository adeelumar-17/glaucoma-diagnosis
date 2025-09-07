import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os

# -------------------------------
# Enhancement Functions
# -------------------------------
def gamma_correction(image, gamma=0.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def hist_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def enhance_combined(image, gamma=0.5):
    gamma_img = gamma_correction(image, gamma)
    return hist_equalization(gamma_img)

def negative_transform(image):
    return 255 - image

def log_transform(image, c=1):
    img_float = image.astype(np.float32) / 255.0
    log_img = c * np.log1p(img_float)
    log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
    return log_img.astype(np.uint8)

def clahe_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

# -------------------------------
# Plot function
# -------------------------------
def plot_comparison(original, enhanced, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(enhanced, cmap='gray')
    axes[0].set_title(title)
    axes[0].axis("off")

    axes[1].hist(enhanced.ravel(), bins=256, color='black', density=True)
    axes[1].set_title(f"{title} Histogram")

    st.pyplot(fig)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🖼️ Image Enhancement App")
st.write("Upload an image and apply different enhancement techniques with histograms.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

    # Ensure RGB format
    if len(img.shape) == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show original image + histogram
    st.subheader("Original Image")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].hist(img.ravel(), bins=256, color="black", density=True)
    axes[1].set_title("Original Histogram")
    st.pyplot(fig)

    # Technique selection
    st.subheader("Enhancement Techniques")
    gamma_value = st.slider("Gamma Value", 0.1, 3.0, 0.5, 0.1)
    technique = st.selectbox(
        "Choose Technique",
        ["Gamma Correction", "Histogram Equalization", "Combined (Gamma+HistEq)",
         "Negative", "Log Transform", "CLAHE"]
    )

    # Apply enhancement
    if st.button("Apply Enhancement"):
        if technique == "Gamma Correction":
            enhanced = gamma_correction(img, gamma_value)
        elif technique == "Histogram Equalization":
            enhanced = hist_equalization(img)
        elif technique == "Combined (Gamma+HistEq)":
            enhanced = enhance_combined(img, gamma_value)
        elif technique == "Negative":
            enhanced = negative_transform(img)
        elif technique == "Log Transform":
            enhanced = log_transform(img)
        elif technique == "CLAHE":
            enhanced = clahe_equalization(img)
        else:
            enhanced = img

        plot_comparison(img, enhanced, technique)
