import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.colors as mcolors
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Analizador de Pavimento")

# Estética de botones y contenedores
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; font-weight: bold; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES TÉCNICAS ---
S_MIN_COLOR, V_MIN_COLOR = 0.25, 0.15
BLUE_H_MIN, BLUE_H_MAX = 200.0/360.0, 260.0/360.0
ANG_TOL = 20.0
# Factores de conversión px a metros
S_TRANSVERSAL, S_LONGITUDINAL = 0.005368, 0.13644
# Configuración de geometría de calzada
CENTER_OFFSET, PINK_LEFT_OFFSET, PINK_RIGHT_OFFSET = 220, 340, 620

COLOR_MAP = {
    "longitudinal": (255, 0, 0, 160),      # Rojo
    "transversal": (0, 102, 255, 160),     # Azul
    "en_el_eje": (0, 180, 0, 180),         # Verde
    "en_todas_direcciones": (200, 0, 200, 160) # Púrpura
}

# --- FUNCIONES DE PROCESAMIENTO ---

def yellow_mask_rgb_hsv(arr):
    rgb = arr.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (V >= V_MIN_COLOR) & (S >= S_MIN_COLOR) & ~((H >= BLUE_H_MIN) & (H <= BLUE_H_MAX))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    return ndi.binary_opening(mask, structure=np.ones((2, 2)))

def classify_component(xs, ys, W):
    x_c, y_c = xs.mean(), ys.mean()
    Xpx = np.vstack([xs - x_c, ys - y_c])
    evals, evecs = np.linalg.eig(np.cov(Xpx))
    major = evecs[:, np.argmax(evals)]
    ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
    
    # Determinación de zona
    x_center = (W / 2.0) + CENTER_OFFSET
    x1, x3 = int(x_center - PINK_LEFT_OFFSET), int(x_center + PINK_RIGHT_OFFSET)
    x2 = int((x1 + x3) / 2.0)
    
    if x_c < x1: zona = "Berma Izq"
    elif x_c < x2: zona = "Pista 2"
    elif x_c < x3: zona = "Pista 1"
    else: zona = "Berma Der"
    
    # Clasificación por ángulo
    if abs(ang - 90.0) <= ANG_TOL: clase = "longitudinal"
    elif (ang <= ANG_TOL) or (ang >= 180.0 - ANG_TOL): clase = "transversal"
    else: clase = "en_todas_direcciones"
    
    Lpx = float(np.ptp(major[0]*(xs-x_c) + major[1]*(ys-y_c)))
    return clase, Lpx, zona, x_c, y_c

def annotate_image(img_pil, labels, df_comp):
    base = img_pil.convert("RGBA")
    W, H = base.size
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    ov_arr = np.array(overlay)
    
    # Pintar las grietas detectadas
    for _, r in df_comp.iterrows():
        comp_mask = (labels == r["id"])
        ov_arr[comp_mask] = COLOR_MAP.get(r["clase"], (0,0,0,160))
    
    combined = Image.alpha_composite(base, Image.fromarray(ov_arr))
    draw = ImageDraw.Draw(combined)
    
    # Cargar fuente externa para etiquetas legibles
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except:
        font = ImageFont.load_default()

    # Dibujar etiquetas de texto
    for _, r in df_comp.iterrows():
        txt = f"{r['clase'].capitalize()}: {r['metros']}m"
        draw.text((r['x'], r['y']), txt, fill="white", font=font, stroke_width=2, stroke_fill="black")
        
    return combined.convert("RGB")

# --- LÓGICA DE NAVEGACIÓN ---

if 'img_orig' not in st.session_state:
    st.session_state.img_orig = None
    st.session_state.img_proc = None
    st.session_state.resumen = None

# PANEL LATERAL (Diseñado para la carga desde iPad)
with st.sidebar:
    st.header("📸 Entrada de Datos")
    st.write("Cargue la monografía capturada para procesar.")
    file = st.file_uploader("Seleccionar imagen", type=["jpg", "jpeg", "png"])
    
    if file and st.button("🚀 PROCESAR AHORA"):
        with st.spinner('Analizando imagen...'):
            img = Image.open(file).convert("RGB")
            arr = np.array(img)
            
            # Procesamiento core
            mask = yellow_mask_rgb_hsv(arr)
            labels, ncomp = ndi.label(mask)
            slices = ndi.find_objects(labels)
            
            rows = []
            for i, slc in enumerate(slices, start=1):
                if slc is None or np.sum(labels[slc]==i) < 80: continue
                ys, xs = np.nonzero(labels[slc]==i)
                clase, Lpx, zona, xc, yc = classify_component(xs+slc[1].start, ys+slc[0].start, arr.shape[1])
                
                # Conversión a metros reales
                factor = S_TRANSVERSAL if clase == "transversal" else S_LONGITUDINAL
                metros = round(Lpx * factor, 2)
                
                rows.append({"id": i, "clase": clase, "metros": metros, "zona": zona, "x":
