import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage import color
import scipy.ndimage
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Enhancement Pipeline", layout="centered")
st.title("Image Enhancement with White Balance, Dehazing, and CLAHE")

# ------------------- Utility Functions ------------------- #

def guided_filter(I, p, r, eps):
    size = (2 * r + 1, 2 * r + 1)
    mean_I = scipy.ndimage.uniform_filter(I, size=size)
    mean_p = scipy.ndimage.uniform_filter(p, size=size)
    mean_Ip = scipy.ndimage.uniform_filter(I * p, size=size)
    mean_II = scipy.ndimage.uniform_filter(I * I, size=size)
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = scipy.ndimage.uniform_filter(a, size=size)
    mean_b = scipy.ndimage.uniform_filter(b, size=size)
    return mean_a * I + mean_b

def white_balance_gray_world(img):
    img_float = img.astype(np.float32)
    original_avg = img_float.mean(axis=(0, 1))
    gray = original_avg.mean()
    gain = gray / np.maximum(original_avg, 1e-8)
    balanced = img_float * gain[np.newaxis, np.newaxis, :]
    balanced_clipped = np.clip(balanced, 0, 255)
    return balanced_clipped.astype(np.uint8)

def dark_channel_prior_dehaze(img, patch_size=15, omega=0.95, t0=0.1, gf_radius=40, gf_eps=1e-3):
    img_float = img.astype(np.float32) / 255.0
    H, W, _ = img_float.shape
    min_channel = np.min(img_float, axis=2)
    dark = scipy.ndimage.minimum_filter(min_channel, size=patch_size)
    A = np.max(img_float[dark >= np.percentile(dark, 99.9)], axis=0)
    A = np.clip(A, 0, 1.0)
    A_broadcast = A[np.newaxis, np.newaxis, :]
    img_norm = img_float / np.maximum(A_broadcast, 1e-8)
    dark_norm = np.min(img_norm, axis=2)
    trans_raw = 1 - omega * dark_norm
    trans_raw = np.clip(trans_raw, t0, 1.0)
    gray_img = color.rgb2gray(img_float)
    trans_refined = guided_filter(gray_img, trans_raw, gf_radius, gf_eps)
    trans_refined = np.clip(trans_refined, t0, 1.0)[..., np.newaxis]
    J = (img_float - A_broadcast) / np.maximum(trans_refined, 1e-8) + A_broadcast
    return np.clip(J * 255.0, 0, 255).astype(np.uint8)

def clahe_contrast_enhancement(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L)
    enhanced_lab = cv2.merge((L_clahe, A, B))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_bgr

# ------------------- Streamlit UI ------------------- #

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img_pil)

    st.subheader("Original Image")
    st.image(img_np, caption="Original", width=img_np.shape[1], use_container_width=False)

    # Step 1: White Balance
    wb_img = white_balance_gray_world(img_np)
    st.subheader("White Balanced")
    st.image(wb_img, caption="White Balanced", width=wb_img.shape[1], use_container_width=False)

    # Step 2: Dehazing
    dehazed_img = dark_channel_prior_dehaze(wb_img)
    st.subheader("Dehazed Image")
    st.image(dehazed_img, caption="Dehazed", width=dehazed_img.shape[1], use_container_width=False)

    # Step 3: CLAHE
    enhanced_img = clahe_contrast_enhancement(dehazed_img)
    st.subheader("Final Enhanced Image (CLAHE)")
    st.image(enhanced_img, caption="Enhanced", width=enhanced_img.shape[1], use_container_width=False)

    # Download button
    final_pil = Image.fromarray(enhanced_img)
    st.download_button("Download Enhanced Image", data=final_pil.tobytes(), file_name="enhanced.png", mime="image/png")
