import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.colors as mcolors
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Analizador MEL - Digitalización")

# --- CONSTANTES TÉCNICAS (Tu código original) ---
ANCHO_BERMA = 2.0
ANCHO_PISTA = 3.5
ANCHO_TOTAL = ANCHO_BERMA + ANCHO_PISTA + ANCHO_PISTA + ANCHO_BERMA # 11 m
ANG_TOL = 20.0
EJE_TOL_M = 0.2
MIN_YELLOW_PIXELS = 150
MIN_YELLOW_RATIO = 0.0005
S_MIN_COLOR = 0.25
V_MIN_COLOR = 0.15
BLUE_H_MIN = 200.0 / 360.0
BLUE_H_MAX = 260.0 / 360.0

COLOR_MAP = {
    "longitudinal": (255, 0, 0, 160),
    "transversal": (0, 102, 255, 160),
    "en_el_eje": (0, 180, 0, 180),
    "en_todas_direcciones": (200, 0, 200, 160)
}

S_TRANSVERSAL = 0.005368
S_LONGITUDINAL = 0.13644
S_TODAS = 0.13655

CENTER_OFFSET = 220
PINK_LEFT_OFFSET = 340
PINK_RIGHT_OFFSET = 620

HUELLA_P2_LEFT_OFFSET, HUELLA_P2_LEFT_WIDTH = -100, 100
HUELLA_P2_RIGHT_OFFSET, HUELLA_P2_RIGHT_WIDTH = 100, 85
HUELLA_P1_LEFT_OFFSET, HUELLA_P1_LEFT_WIDTH = -120, 100
HUELLA_P1_RIGHT_OFFSET, HUELLA_P1_RIGHT_WIDTH = 120, 100
HUELLA_MIN_FRAC = 0.75

# --- FUNCIONES DE MOTOR ORIGINAL (Sin cambios en lógica) ---

def yellow_mask_rgb_hsv(arr):
    rgb = arr.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (V >= V_MIN_COLOR) & (S >= S_MIN_COLOR) & ~((H >= BLUE_H_MIN) & (H <= BLUE_H_MAX))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    return mask

def compute_zone_bounds(W):
    x_center_shifted = (W / 2.0) + CENTER_OFFSET
    x1 = int(round(max(0, min(W, x_center_shifted - PINK_LEFT_OFFSET))))
    x3 = int(round(max(0, min(W, x_center_shifted + PINK_RIGHT_OFFSET))))
    x2 = int(round((x1 + x3) / 2.0))
    
    def get_h(center_p, off, width, x_min, x_max):
        c = center_p + off
        return int(round(max(x_min, min(x_max, c - width/2.0)))), int(round(max(x_min, min(x_max, c + width/2.0))))

    h_p2 = (*get_h((x1+x2)/2.0, HUELLA_P2_LEFT_OFFSET, HUELLA_P2_LEFT_WIDTH, x1, x2),
            *get_h((x1+x2)/2.0, HUELLA_P2_RIGHT_OFFSET, HUELLA_P2_RIGHT_WIDTH, x1, x2))
    h_p1 = (*get_h((x2+x3)/2.0, HUELLA_P1_LEFT_OFFSET, HUELLA_P1_LEFT_WIDTH, x2, x3),
            *get_h((x2+x3)/2.0, HUELLA_P1_RIGHT_OFFSET, HUELLA_P1_RIGHT_WIDTH, x2, x3))
    return (0, x1, x2, x3, W), h_p2, h_p1

def classify_component(xs, ys, W):
    x_c, y_c = xs.mean(), ys.mean()
    Xpx = np.vstack([xs - x_c, ys - y_c])
    evals, evecs = np.linalg.eig(np.cov(Xpx))
    major = evecs[:, np.argmax(evals)]
    ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
    
    (x0, x1, x2, x3, x4), _, _ = compute_zone_bounds(W)
    zona = "Berma Izq" if x_c < x1 else "P2" if x_c < x2 else "P1" if x_c < x3 else "Berma Der"
    
    # Lógica de distancia al eje de tu código
    x_p2_eje = x1 + (ANCHO_PISTA) * ((x3-x1)/7.0) # Simplificación de tu px_per_m
    dist_min_m = float(np.min(np.abs(xs - x_p2_eje))) * S_LONGITUDINAL

    if abs(ang - 90.0) <= ANG_TOL:
        clase = "en_el_eje" if dist_min_m <= EJE_TOL_M else "longitudinal"
    elif (ang <= ANG_TOL) or (ang >= 180.0 - ANG_TOL):
        clase = "transversal"
    else:
        clase = "en_todas_direcciones"
    
    Lpx = float(np.ptp(major[0]*(xs-x_c) + major[1]*(ys-y_c)))
    return clase, Lpx, zona, x_c, y_c

def build_zone_maps(W, H):
    full = np.full((H, W), -1, dtype=np.int8)
    (x0, x1, x2, x3, x4), hp2, hp1 = compute_zone_bounds(W)
    full[:, x0:x1], full[:, x1:x2], full[:, x2:x3], full[:, x3:x4] = 0, 1, 2, 3
    if hp2[1]>hp2[0]: full[:, hp2[0]:hp2[1]] = 4
    if hp2[3]>hp2[2]: full[:, hp2[2]:hp2[3]] = 4
    if hp1[1]>hp1[0]: full[:, hp1[0]:hp1[1]] = 5
    if hp1[3]>hp1[2]: full[:, hp1[2]:hp1[3]] = 5
    return full

def annotate_image(img_pil, labels, df_comp):
    base = img_pil.convert("RGBA")
    W, H = base.size
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    ov_arr = np.array(overlay)
    for _, r in df_comp.iterrows():
        ov_arr[labels == r["id"]] = COLOR_MAP.get(r["clase"], (0,0,0,160))
    
    combined = Image.alpha_composite(base, Image.fromarray(ov_arr))
    draw = ImageDraw.Draw(combined)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 14) # Tamaño 14 solicitado
    except: font = ImageFont.load_default()

    for _, r in df_comp.iterrows():
        label = f"{r['clase'].replace('_',' ')} | {r['metros']:.2f} m"
        draw.text((r['x'], r['y']), label, fill="white", font=font, stroke_width=1, stroke_fill="black")
    return combined.convert("RGB")

# --- APP FLOW ---

if 'img_orig' not in st.session_state:
    st.session_state.update({'img_orig': None, 'img_proc': None, 'resumen': None})

st.sidebar.title("Navegación")
app_mode = st.sidebar.radio("Seleccione Pantalla", ["Carga (iPad)", "Visualización Pública"])

if app_mode == "Carga (iPad)":
    st.title("📤 Carga de Monografía")
    uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and st.button("🚀 INICIAR PROCESAMIENTO"):
        with st.spinner('Ejecutando motor original...'):
            img = Image.open(uploaded_file).convert("RGB")
            arr = np.array(img)
            H, W = arr.shape[:2]
            mask = yellow_mask_rgb_hsv(arr)
            
            labels, ncomp = ndi.label(mask)
            slices = ndi.find_objects(labels)
            z_map = build_zone_maps(W, H)
            
            comp_data, prorr = [], []
            for i, slc in enumerate(slices, start=1):
                if slc is None or np.sum(labels[slc]==i) < 80: continue
                ys, xs = np.nonzero(labels[slc]==i)
                clase, Lpx, zona, xc, yc = classify_component(xs+slc[1].start, ys+slc[0].start, W)
                
                factor = S_TRANSVERSAL if clase=="transversal" else S_LONGITUDINAL if clase in ["longitudinal","en_el_eje"] else S_TODAS
                Lm = Lpx * factor
                comp_data.append({"id":i, "clase":clase, "metros":Lm, "x":xc, "y":yc})
                
                # Lógica de Prorrateo de tu código
                c_mask = (labels == i)
                total_px = np.sum(c_mask)
                f_h2 = np.sum(z_map[c_mask] == 4) / total_px
                f_h1 = np.sum(z_map[c_mask] == 5) / total_px
                
                if f_h2 >= HUELLA_MIN_FRAC: prorr.append({"Zona": "Huella P2", "Clase": clase, "Metros": Lm})
                elif f_h1 >= HUELLA_MIN_FRAC: prorr.append({"Zona": "Huella P1", "Clase": clase, "Metros": Lm})
                else: prorr.append({"Zona": zona, "Clase": clase, "Metros": Lm})

            df_res = pd.DataFrame(prorr).groupby(["Zona", "Clase"])["Metros"].sum().reset_index()
            st.session_state.update({'img_orig': img, 'img_proc': annotate_image(img, labels, pd.DataFrame(comp_data)), 'resumen': df_res})
            st.success("¡Procesado! Cambie a Visualización Pública.")

else:
    st.title("📊 Resultados en Pantalla")
    if st.session_state.img_proc:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(st.session_state.img_orig, use_container_width=True)
        with col2:
            st.subheader("Clasificada (Prórrateo Huellas)")
            st.image(st.session_state.img_proc, use_container_width=True)
        
        st.divider()
        st.subheader("📋 Resumen de Mediciones")
        st.table(st.session_state.resumen)
        
        if st.button("🆕 NUEVA INSPECCIÓN"):
            for key in ['img_orig', 'img_proc', 'resumen']: st.session_state[key] = None
            st.rerun()
    else:
        st.info("Esperando datos desde el iPad...")
