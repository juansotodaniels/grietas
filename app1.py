import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.colors as mcolors
import io
import base64

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Analizador de Pavimentos - Escala Real 2550px")

# --- CONSTANTES TÉCNICAS FIJAS ---
W_OBJETIVO = 2550  # Forzamos el ancho de diseño de Paint
X_CENTER_OBJETIVO = 1515  # El eje que definiste

# Calculamos el OFFSET necesario: 1515 = (2550 / 2) + OFFSET -> OFFSET = 240
CENTER_OFFSET = X_CENTER_OBJETIVO - (W_OBJETIVO / 2.0) 

# Mantengo tus proporciones de zonas para esta escala
PINK_LEFT_OFFSET = 340 
PINK_RIGHT_OFFSET = 620 

ANG_TOL, EJE_TOL_M = 20.0, 0.2
HUELLA_MIN_FRAC = 0.75

# ESCALAS CALIBRADAS
S_LONGITUDINAL = 0.075196
S_TRANSVERSAL = 0.007338
S_TODAS = 0.036909

COLOR_MAP = {
    "Longitudinal": (255, 0, 0, 160),
    "Transverse": (0, 102, 255, 160),
    "On Axis": (0, 180, 0, 180),
    "In all directions": (200, 0, 200, 160)
}

# --- FUNCIONES DE PROCESAMIENTO ---
def yellow_mask_rgb_hsv(arr):
    rgb = arr.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(rgb)
    mask = ((hsv[..., 2] < 0.35) | ((hsv[..., 1] > 0.25) & (hsv[..., 2] > 0.40)))
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndi.binary_closing(mask, structure=np.ones((4, 4)))
    return mask

def compute_zone_bounds(W):
    x_center_shifted = (W / 2.0) + CENTER_OFFSET
    x1 = int(round(max(0, min(W, x_center_shifted - PINK_LEFT_OFFSET))))
    x3 = int(round(max(0, min(W, x_center_shifted + PINK_RIGHT_OFFSET))))
    x2 = int(round(x_center_shifted)) # El eje divisorio ES el centro ajustado
    return (0, x1, x2, x3, W)

def classify_component(xs, ys, W):
    x_c, y_c = xs.mean(), ys.mean()
    Xpx = np.vstack([xs - x_c, ys - y_c])
    cov = np.cov(Xpx)
    if cov.ndim < 2: return "Unknown", 0, "Outside", x_c, y_c
    evals, evecs = np.linalg.eig(cov)
    major = evecs[:, np.argmax(evals)]
    ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
    
    bounds = compute_zone_bounds(W)
    if x_c < bounds[1]: zona = "Left Shoulder"
    elif x_c < bounds[2]: zona = "Lane 2"
    elif x_c < bounds[3]: zona = "Lane 1"
    else: zona = "Right Shoulder"
    
    # Eje On Axis = Centro Ajustado
    x_p2_eje = (W / 2.0) + CENTER_OFFSET
    dist_min_m = float(np.min(np.abs(xs - x_p2_eje))) * S_LONGITUDINAL

    if abs(ang - 90.0) <= ANG_TOL:
        clase = "On Axis" if dist_min_m <= EJE_TOL_M else "Longitudinal"
    elif (ang <= ANG_TOL) or (ang >= 180.0 - ANG_TOL):
        clase = "Transverse"
    else:
        clase = "In all directions"
    
    Lpx = float(np.ptp(major[0]*(xs-x_c) + major[1]*(ys-y_c)))
    return clase, Lpx, zona, x_c, y_c

def annotate_image_final(img_pil, labels, df_comp):
    base = img_pil.convert("RGBA")
    W, H = base.size
    zone_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    zdraw = ImageDraw.Draw(zone_ov)
    x0, x1, x2, x3, x4 = compute_zone_bounds(W)
    
    cols = {"SHOULDER": (255,255,0,70), "LANE2": (0,200,255,70), "LANE1": (255,150,255,70)}
    zdraw.rectangle([x0,0,x1,H], fill=cols["SHOULDER"])
    zdraw.rectangle([x1,0,x2,H], fill=cols["LANE2"])
    zdraw.rectangle([x2,0,x3,H], fill=cols["LANE1"])
    zdraw.rectangle([x3,0,x4,H], fill=cols["SHOULDER"])
    
    crack_ov = np.zeros((H, W, 4), dtype=np.uint8)
    for _, r in df_comp.iterrows():
        crack_ov[labels == r["id"]] = COLOR_MAP.get(r["clase"], (0,0,0,160))
    
    out = Image.alpha_composite(base, zone_ov)
    out = Image.alpha_composite(out, Image.fromarray(crack_ov))
    draw = ImageDraw.Draw(out)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 25)
    except: font = ImageFont.load_default()

    for _, r in df_comp.iterrows():
        txt = f"{r['clase']}: {r['metros']:.2f}m"
        draw.text((r['x'], r['y']), txt, fill=(0,0,0,255), font=font)
    return out.convert("RGB")

# --- APP ---
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🚀 Analizador de Pavimentos (Auto-Rescale 2550px)")
up = st.file_uploader("Subir imagen de terreno", type=["jpg", "png", "jpeg"])

if up and st.button("PROCESAR"):
    with st.spinner('Re-escalando y analizando...'):
        # 1. CARGA Y REDIMENSIONAMIENTO AUTOMÁTICO
        img_raw = Image.open(up).convert("RGB")
        # Calculamos el nuevo alto manteniendo la relación de aspecto
        aspect_ratio = img_raw.size[1] / img_raw.size[0]
        h_objetivo = int(W_OBJETIVO * aspect_ratio)
        img = img_raw.resize((W_OBJETIVO, h_objetivo), Image.Resampling.LANCZOS)
        
        arr = np.array(img)
        mask = yellow_mask_rgb_hsv(arr)
        labels, ncomp = ndi.label(mask)
        slices = ndi.find_objects(labels)
        
        rows, prorr = [], []
        for i, slc in enumerate(slices, start=1):
            if slc is None or np.sum(labels[slc]==i) < 250: continue
            ys, xs = np.nonzero(labels[slc]==i)
            clase, Lpx, zona, xc, yc = classify_component(xs+slc[1].start, ys+slc[0].start, W_OBJETIVO)
            
            f = S_TRANSVERSAL if clase == "Transverse" else (S_LONGITUDINAL if clase in ["Longitudinal", "On Axis"] else S_TODAS)
            Lm = Lpx * f
            rows.append({"id":i, "clase":clase, "metros":Lm, "x":xc, "y":yc})
            prorr.append({"Zone": zona, "Type": clase, "Meters": Lm})

        df_p = pd.DataFrame(prorr).groupby(["Zone", "Type"])["Meters"].sum().reset_index() if prorr else pd.DataFrame()

        st.session_state.data = {
            "orig": img, 
            "proc": annotate_image_final(img, labels, pd.DataFrame(rows)) if rows else img, 
            "res": df_p
        }
        st.success(f"Imagen procesada a {W_OBJETIVO}px. Eje fijado en {X_CENTER_OBJETIVO}px.")

if st.session_state.data:
    c1, c2 = st.columns(2)
    c1.image(st.session_state.data["orig"], caption="Imagen Re-escalada (2550px)")
    c2.image(st.session_state.data["proc"], caption="Mapa de Patologías")
    st.dataframe(st.session_state.data["res"], use_container_width=True)
