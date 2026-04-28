import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.colors as mcolors
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Analizador de Pavimento")

# Estética de la interfaz
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES TÉCNICAS ---
S_MIN_COLOR, V_MIN_COLOR = 0.25, 0.15
BLUE_H_MIN, BLUE_H_MAX = 200.0/360.0, 260.0/360.0
ANG_TOL = 20.0
S_TRANSVERSAL, S_LONGITUDINAL = 0.005368, 0.13644
CENTER_OFFSET, PINK_LEFT_OFFSET, PINK_RIGHT_OFFSET = 220, 340, 620

COLOR_MAP = {
    "longitudinal": (255, 0, 0, 160),
    "transversal": (0, 102, 255, 160),
    "en_el_eje": (0, 180, 0, 180),
    "en_todas_direcciones": (200, 0, 200, 160)
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
    
    x_center = (W / 2.0) + CENTER_OFFSET
    x1, x3 = int(x_center - PINK_LEFT_OFFSET), int(x_center + PINK_RIGHT_OFFSET)
    x2 = int((x1 + x3) / 2.0)
    
    if x_c < x1: zona = "Berma Izq"
    elif x_c < x2: zona = "Pista 2"
    elif x_c < x3: zona = "Pista 1"
    else: zona = "Berma Der"
    
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
    
    for _, r in df_comp.iterrows():
        comp_mask = (labels == r["id"])
        ov_arr[comp_mask] = COLOR_MAP.get(r["clase"], (0,0,0,160))
    
    combined = Image.alpha_composite(base, Image.fromarray(ov_arr))
    draw = ImageDraw.Draw(combined)
    
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except:
        font = ImageFont.load_default()

    for _, r in df_comp.iterrows():
        txt = f"{r['clase']}: {r['metros']}m"
        draw.text((r['x'], r['y']), txt, fill="white", font=font, stroke_width=2, stroke_fill="black")
        
    return combined.convert("RGB")

# --- LÓGICA DE LA APLICACIÓN ---

if 'img_orig' not in st.session_state:
    st.session_state.img_orig = None
    st.session_state.img_proc = None
    st.session_state.resumen = None

# Sidebar para el iPad
with st.sidebar:
    st.header("📸 Entrada iPad")
    file = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])
    if file and st.button("🚀 PROCESAR"):
        with st.spinner('Procesando...'):
            img = Image.open(file).convert("RGB")
            arr = np.array(img)
            mask = yellow_mask_rgb_hsv(arr)
            labels, ncomp = ndi.label(mask)
            slices = ndi.find_objects(labels)
            
            rows = []
            for i, slc in enumerate(slices, start=1):
                if slc is None or np.sum(labels[slc]==i) < 80:
                    continue
                ys, xs = np.nonzero(labels[slc]==i)
                clase, Lpx, zona, xc, yc = classify_component(xs+slc[1].start, ys+slc[0].start, arr.shape[1])
                factor = S_TRANSVERSAL if clase == "transversal" else S_LONGITUDINAL
                metros = round(Lpx * factor, 2)
                # Línea corregida:
                rows.append({"id": i, "clase": clase, "metros": metros, "zona": zona, "x": xc, "y": yc})
            
            df = pd.DataFrame(rows)
            st.session_state.img_orig = img
            st.session_state.img_proc = annotate_image(img, labels, df)
            if not df.empty:
                st.session_state.resumen = df.groupby(['zona', 'clase'])['metros'].sum().reset_index()
            else:
                st.session_state.resumen = pd.DataFrame(columns=['zona', 'clase', 'metros'])
            st.rerun()

# Pantalla Pública
st.title("🚧 Sistema de Inspección Vial")

if st.session_state.img_proc:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(st.session_state.img_orig, use_container_width=True)
    with c2:
        st.subheader("Procesada")
        st.image(st.session_state.img_proc, use_container_width=True)
    
    st.divider()
    st.subheader("📊 Tabla de Resultados")
    st.table(st.session_state.resumen)
    
    if st.button("🆕 EMPEZAR DE NUEVO"):
        st.session_state.img_orig = None
        st.session_state.img_proc = None
        st.session_state.resumen = None
        st.rerun()
else:
    st.info("Esperando carga desde el iPad...")
