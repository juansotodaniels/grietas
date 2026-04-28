import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.colors as mcolors
import io

# --- CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(layout="wide", page_title="Analizador de Pavimento")

# Estilo para ocultar el menú de carga en la vista pública si se desea, 
# pero lo mantendremos en el sidebar para el iPad.
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTES TÉCNICAS (Tu lógica original) ---
PK_DEFAULT = (119.500, 119.750)
ANCHO_TOTAL = 11.0 # 2.0 + 3.5 + 3.5 + 2.0
S_MIN_COLOR, V_MIN_COLOR = 0.25, 0.15
BLUE_H_MIN, BLUE_H_MAX = 200.0 / 360.0, 260.0 / 360.0
ANG_TOL, EJE_TOL_M = 20.0, 0.2
HUELLA_MIN_FRAC = 0.75

COLOR_MAP = {
    "longitudinal": (255, 0, 0, 160),
    "transversal": (0, 102, 255, 160),
    "en_el_eje": (0, 180, 0, 180),
    "en_todas_direcciones": (200, 0, 200, 160)
}

S_TRANSVERSAL, S_LONGITUDINAL, S_TODAS = 0.005368, 0.13644, 0.13655
CENTER_OFFSET, PINK_LEFT_OFFSET, PINK_RIGHT_OFFSET = 220, 340, 620

# Configuración Huellas
HUELLA_P2_LEFT_OFFSET, HUELLA_P2_LEFT_WIDTH = -100, 100
HUELLA_P2_RIGHT_OFFSET, HUELLA_P2_RIGHT_WIDTH = 100, 85
HUELLA_P1_LEFT_OFFSET, HUELLA_P1_LEFT_WIDTH = -120, 100
HUELLA_P1_RIGHT_OFFSET, HUELLA_P1_RIGHT_WIDTH = 120, 100

# --- FUNCIONES DE PROCESAMIENTO (Adaptadas para memoria) ---

def yellow_mask_rgb_hsv(arr):
    rgb = arr.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (V >= V_MIN_COLOR) & (S >= S_MIN_COLOR) & ~((H >= BLUE_H_MIN) & (H <= BLUE_H_MAX))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    return ndi.binary_opening(mask, structure=np.ones((2, 2)))

def compute_zone_bounds(W):
    x_center = (W / 2.0) + CENTER_OFFSET
    x1, x3 = int(x_center - PINK_LEFT_OFFSET), int(x_center + PINK_RIGHT_OFFSET)
    x1, x3 = max(0, min(W, x1)), max(0, min(W, x3))
    x2 = int((x1 + x3) / 2.0)
    
    def get_h(center_p, off, width, x_min, x_max):
        c = center_p + off
        return int(max(x_min, min(x_max, c - width/2))), int(max(x_min, min(x_max, c + width/2)))

    h_p2 = (*get_h((x1+x2)/2, HUELLA_P2_LEFT_OFFSET, HUELLA_P2_LEFT_WIDTH, x1, x2),
            *get_h((x1+x2)/2, HUELLA_P2_RIGHT_OFFSET, HUELLA_P2_RIGHT_WIDTH, x1, x2))
    h_p1 = (*get_h((x2+x3)/2, HUELLA_P1_LEFT_OFFSET, HUELLA_P1_LEFT_WIDTH, x2, x3),
            *get_h((x2+x3)/2, HUELLA_P1_RIGHT_OFFSET, HUELLA_P1_RIGHT_WIDTH, x2, x3))
    return (0, x1, x2, x3, W), h_p2, h_p1

def classify_component(xs, ys, W, x_left_dummy=0, x_right_dummy=1000):
    x_c, y_c = xs.mean(), ys.mean()
    Xpx = np.vstack([xs - x_c, ys - y_c])
    evals, evecs = np.linalg.eig(np.cov(Xpx))
    major = evecs[:, np.argmax(evals)]
    ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
    
    (x0, x1, x2, x3, x4), _, _ = compute_zone_bounds(W)
    zona = "Berma Izq" if x_c < x1 else "P2" if x_c < x2 else "P1" if x_c < x3 else "Berma Der"
    
    if abs(ang - 90.0) <= ANG_TOL:
        clase = "longitudinal" # Simplificado para el prototipo
    elif (ang <= ANG_TOL) or (ang >= 180.0 - ANG_TOL):
        clase = "transversal"
    else:
        clase = "en_todas_direcciones"
    
    return clase, float(np.ptp(major[0]*(xs-x_c) + major[1]*(ys-y_c))), zona, x_c, y_c

def annotate_image(img_pil, labels, df_comp):
    base = img_pil.convert("RGBA")
    W, H = base.size
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    
    # Dibujar zonas
    z, h2, h1 = compute_zone_bounds(W)
    draw.rectangle([z[1], 0, z[2], H], fill=(0, 200, 255, 40)) # P2
    draw.rectangle([z[2], 0, z[3], H], fill=(255, 150, 255, 40)) # P1
    
    # Dibujar grietas
    ov_arr = np.array(overlay)
    for _, r in df_comp.iterrows():
        comp_mask = (labels == r["id"])
        ov_arr[comp_mask] = COLOR_MAP.get(r["clase"], (0,0,0,160))
    
    combined = Image.alpha_composite(base, Image.fromarray(ov_arr))
    return combined.convert("RGB")

# --- LÓGICA DE LA APP ---

if 'estado' not in st.session_state:
    st.session_state.estado = 'inicio'
    st.session_state.img_orig = None
    st.session_state.img_proc = None
    st.session_state.resultados = None

# SIDEBAR: IPAD
with st.sidebar:
    st.header("📲 Entrada iPad")
    uploaded_file = st.file_uploader("Cargar Monografía", type=["jpg", "png", "jpeg"])
    if uploaded_file and st.button("⚙️ Procesar"):
        with st.spinner('Analizando pavimento...'):
            # Proceso
            img_orig = Image.open(uploaded_file).convert("RGB")
            arr = np.array(img_orig)
            mask = yellow_mask_rgb_hsv(arr)
            labels, ncomp = ndi.label(mask)
            slices = ndi.find_objects(labels)
            
            comp_rows = []
            for i, slc in enumerate(slices, start=1):
                if slc is None or np.sum(labels[slc]==i) < 80: continue
                ys, xs = np.nonzero(labels[slc]==i)
                clase, Lpx, zona, xc, yc = classify_component(xs+slc[1].start, ys+slc[0].start, arr.shape[1])
                Lm = Lpx * (S_TRANSVERSAL if clase=="transversal" else S_LONGITUDINAL)
                comp_rows.append({"id": i, "clase": clase, "metros": round(Lm, 2), "zona": zona})
            
            df = pd.DataFrame(comp_rows)
            st.session_state.img_orig = img_orig
            st.session_state.img_proc = annotate_image(img_orig, labels, df)
            
            # Resumen para la tabla
            resumen = df.groupby(['zona', 'clase'])['metros'].sum().reset_index()
            st.session_state.resultados = resumen
            st.session_state.estado = 'ver'
            st.rerun()

# CUERPO PRINCIPAL: PANTALLA PÚBLICA
st.title("🚧 Sistema de Clasificación de Grietas")

if st.session_state.estado == 'ver':
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(st.session_state.img_orig, use_container_width=True)
    with c2:
        st.subheader("Clasificada")
        st.image(st.session_state.img_proc, use_container_width=True)
    
    st.divider()
    st.subheader("📊 Cuadro de Valores Calculados")
    st.dataframe(st.session_state.resultados, use_container_width=True)
    
    if st.button("🆕 Empezar de nuevo"):
        st.session_state.estado = 'inicio'
        st.rerun()
else:
    st.info("Esperando que el técnico cargue una imagen desde el iPad...")
    # Imagen de ejemplo o instrucciones
    st.write("1. Use el panel lateral en el iPad.\n2. Cargue la foto.\n3. Presione Procesar.")
