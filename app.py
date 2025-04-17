import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage import color
import scipy.ndimage
from io import BytesIO

st.set_page_config(page_title="Image Enhancement Pipeline", layout="wide")
st.title("✨ Image Enhancement with White Balance, Dehazing & CLAHE ✨")


def guided_filter(I, p, r, eps):
    size = (2 * r + 1, 2 * r + 1)
    mean_I = scipy.ndimage.uniform_filter(I, size=size)
    mean_p = scipy.ndimage.uniform_filter(p, size=size)
    cov_Ip = scipy.ndimage.uniform_filter(I * p, size=size) - mean_I * mean_p
    var_I = scipy.ndimage.uniform_filter(I * I, size=size) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = scipy.ndimage.uniform_filter(a, size=size)
    mean_b = scipy.ndimage.uniform_filter(b, size=size)
    return mean_a * I + mean_b


def white_balance_gray_world(img):
    img_f = img.astype(np.float32)
    avg = img_f.mean(axis=(0, 1))
    gray = avg.mean()
    gain = gray / np.maximum(avg, 1e-8)
    out = img_f * gain[np.newaxis, np.newaxis, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def dark_channel_prior_dehaze(img, radius=15, omega=0.95, t0=0.1, gf_r=40, gf_eps=1e-3):
    img_f = img.astype(np.float32) / 255.0
    dark = np.min(img_f, axis=2)
    dark_min = scipy.ndimage.minimum_filter(dark, size=radius)
    A = np.max(img_f[dark_min >= np.percentile(dark_min, 99.9)], axis=0)
    A = np.clip(A, 0, 1)
    norm = img_f / np.maximum(A[np.newaxis, np.newaxis, :], 1e-8)
    dark2 = np.min(norm, axis=2)
    t = 1 - omega * dark2
    t = np.clip(t, t0, 1)
    gray = color.rgb2gray(img_f)
    t_ref = guided_filter(gray, t, gf_r, gf_eps)
    t_ref = np.clip(t_ref, t0, 1)[..., np.newaxis]
    J = (img_f - A[np.newaxis, np.newaxis, :]) / np.maximum(t_ref, 1e-8) + A[np.newaxis, np.newaxis, :]
    return np.clip(J * 255.0, 0, 255).astype(np.uint8)


def clahe_enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge((L2, A, B)), cv2.COLOR_LAB2RGB)


uploaded = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB"))
    st.sidebar.subheader("Original Image")
    st.sidebar.image(img, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    with col1, st.spinner("🔄 White Balancing..."):
        wb = white_balance_gray_world(img)
        st.image(wb, use_container_width=True, caption="White Balanced")

    with col2, st.spinner("🔄 Dehazing..."):
        dh = dark_channel_prior_dehaze(wb)
        st.image(dh, use_container_width=True, caption="Dehazed")

    with col3, st.spinner("🔄 Applying CLAHE..."):
        ce = clahe_enhance(dh)
        st.image(ce, use_container_width=True, caption="Enhanced")

    buf = BytesIO()
    Image.fromarray(ce).save(buf, format="PNG")
    buf.seek(0)
    st.sidebar.download_button("💾 Download Enhanced", data=buf, file_name="enhanced.png", mime="image/png")
