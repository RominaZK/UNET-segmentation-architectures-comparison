import os, glob, io, random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import albumentations as A
from PIL import Image

# =========================
# Configuración general
# =========================
st.set_page_config(page_title="Segmentation Dashboard", page_icon="🩻", layout="wide")

# Carpeta de datos (usa 'assets')
ASSETS = os.environ.get("DASH_ASSETS", "assets")
TPL = "simple_white"

# =========================
# Utilidades
# =========================
def to_uint8(img):
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.max() <= 1.0:
            a *= 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a

def colorize_mask(mask_u8, color=(0,255,0)):
    m = (mask_u8 > 0).astype(np.uint8)
    rgb = np.zeros((*m.shape, 3), dtype=np.uint8)
    rgb[...,0], rgb[...,1], rgb[...,2] = m*color[0], m*color[1], m*color[2]
    return rgb

def overlay(img_u8, mask_u8, color=(0,255,0), alpha=0.45):
    img3 = img_u8 if img_u8.ndim == 3 else np.stack([img_u8]*3, axis=-1)
    mk = colorize_mask(mask_u8, color)
    return (img3.astype(np.float32)*(1-alpha) + mk.astype(np.float32)*alpha).astype(np.uint8)

# =========================
# Carga de datos
# =========================
@st.cache_data
def load_metrics():
    p = os.path.join(ASSETS, "metrics.csv")
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else pd.DataFrame()

@st.cache_data
def load_preview():
    Xp = os.path.join(ASSETS, "X_preview.npy")
    Yp = os.path.join(ASSETS, "Y_preview.npy")
    X = np.load(Xp, allow_pickle=True) if os.path.exists(Xp) else np.empty((0,))
    Y = np.load(Yp, allow_pickle=True) if os.path.exists(Yp) else np.empty((0,))
    return X, Y

@st.cache_data
def load_histories():
    H = {}
    hdir = os.path.join(ASSETS, "histories")
    if os.path.isdir(hdir):
        for f in glob.glob(os.path.join(hdir, "*.csv")):
            name = os.path.splitext(os.path.basename(f))[0]
            try:
                H[name] = pd.read_csv(f)
            except Exception:
                pass
    return H

df = load_metrics()
X_prev, Y_prev = load_preview()
histories = load_histories()

# =========================
# Header
# =========================
st.title("Segmentation Dashboard")

# Tarjetas resumen
if len(df):
    c1,c2,c3,c4 = st.columns(4)
    f = lambda s: "—" if s is None or pd.isna(s) else f"{s:.3f}"
    with c1: st.metric("Best Dice", f(df.get("Dice Mean", pd.Series([np.nan])).max()))
    with c2: st.metric("Best IoU",  f(df.get("IoU Mean",  pd.Series([np.nan])).max()))
    with c3: st.metric("Best mAP",  f(df.get("mAP Mean",  pd.Series([np.nan])).max()))
    with c4: st.metric("Fastest FPS", f(df.get("FPS",     pd.Series([np.nan])).max()))

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics","📈 Loss Curves","🧪 Samples","🔎 EDA"])

# --- Metrics ---
with tab1:
    st.subheader("Model comparison")
    if len(df):
        st.dataframe(df.style.format(precision=4), use_container_width=True)
    else:
        st.info("No se encontró metrics.csv en assets/.")
        