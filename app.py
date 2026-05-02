import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.colors as mcolors
import io
import base64

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Analizador de Pavimentos - Pro")

# --- CONSTANTES TÉCNICAS FIJAS ---
W_OBJETIVO = 2550  
X_CENTER_OBJETIVO = 1515  
CENTER_OFFSET = X_CENTER_OBJETIVO - (W_OBJETIVO / 2.0) 

# Ajuste para simetría total de carriles (3.5m cada uno)
ANCHO_PISTA_PX = 467 
PINK_LEFT_OFFSET = ANCHO_PISTA_PX 
PINK_RIGHT_OFFSET = ANCHO_PISTA_PX 

# Constantes para las franjas rojas (Huellas) simétricas
WP_OFFSET = 110  # Distancia desde el centro del carril
WP_WIDTH = 95    # Grosor de la franja roja

ANG_TOL, EJE_TOL_M = 20.0, 0.2
HUELLA_MIN_FRAC = 0.75

S_LONGITUDINAL = 0.075196
S_TRANSVERSAL = 0.007338
S_TODAS = 0.036909

COLOR_MAP = {
    "Longitudinal": (255, 0, 0, 255),
    "Transverse": (0, 102, 255, 255),
    "On Axis": (0, 180, 0, 255),
    "In all directions": (200, 0, 200, 255)
}

# --- FUNCIONES DE PROCESAMIENTO ---
def adaptive_mask(arr):
    # Detecta cualquier trazo oscuro/colorido sobre fondo claro (Paint)
    gray = np.mean(arr, axis=2)
    mask = gray < 225 
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    return mask

def compute_zone_bounds(W):
    x_center_shifted = (W / 2.0) + CENTER_OFFSET
    x2 = int(round(x_center_shifted))
    x1 = int(round(x2 - PINK_LEFT_OFFSET))
    x3 = int(round(x2 + PINK_RIGHT_OFFSET))
    
    # Cálculo de huellas (franjas rojas) simétricas
    def get_h(cp, off, w):
        return int(round(cp + off - w/2)), int(round(cp + off + w/2))
    
    c_l2 = (x1 + x2) / 2.0
    c_l1 = (x2 + x3) / 2.0
    
    # Cada carril tiene dos huellas
    hp2 = (*get_h(c_l2, -WP_OFFSET, WP_WIDTH), *get_h(c_l2, WP_OFFSET, WP_WIDTH))
    hp1 = (*get_h(c_l1, -WP_OFFSET, WP_WIDTH), *get_h(c_l1, WP_OFFSET, WP_WIDTH))
    
    return (0, x1, x2, x3, W), hp2, hp1

def classify_component(xs, ys, W):
    x_c, y_c = xs.mean(), ys.mean()
    Xpx = np.vstack([xs - x_c, ys - y_c])
    if Xpx.shape[1] < 5: return "In all directions", 0, "Outside", x_c, y_c
    
    cov = np.cov(Xpx)
    evals, evecs = np.linalg.eig(cov)
    idx = np.argmax(evals)
    
    # Si la mancha es muy ancha (no es una línea), es "In all directions"
    if evals[1-idx]/evals[idx] > 0.4:
        clase = "In all directions"
    else:
        major = evecs[:, idx]
        ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
        if abs(ang - 90.0) <= ANG_TOL:
            clase = "Longitudinal"
        elif (ang <= ANG_TOL) or (ang >= 180.0 - ANG_TOL):
            clase = "Transverse"
        else:
            clase = "In all directions"
    
    bounds, _, _ = compute_zone_bounds(W)
    if x_c < bounds[1]: zona = "Left Shoulder"
    elif x_c < bounds[2]: zona = "Lane 2"
    elif x_c < bounds[3]: zona = "Lane 1"
    else: zona = "Right Shoulder"
    
    if clase == "Longitudinal" and abs(x_c - bounds[2]) * S_LONGITUDINAL <= EJE_TOL_M:
        clase = "On Axis"
        
    Lpx = float(np.ptp(xs) if clase=="Transverse" else np.ptp(ys))
    return clase, Lpx, zona, x_c, y_c

def annotate_image_final(img_pil, labels, df_comp):
    base = img_pil.convert("RGBA")
    W, H = base.size
    zone_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    zdraw = ImageDraw.Draw(zone_ov)
    bounds, hp2, hp1 = compute_zone_bounds(W)
    x0, x1, x2, x3, x4 = bounds
    
    # Dibujar Fondos de Carriles
    zdraw.rectangle([x0,0,x1,H], fill=(255,255,0,40)) 
    zdraw.rectangle([x1,0,x2,H], fill=(0,200,255,40)) 
    zdraw.rectangle([x2,0,x3,H], fill=(255,150,255,40))
    zdraw.rectangle([x3,0,x4,H], fill=(255,255,0,40)) 
    
    # Dibujar Huellas Rojas Simétricas
    for h in [hp2, hp1]:
        zdraw.rectangle([h[0],0,h[1],H], fill=(255,0,0,70))
        zdraw.rectangle([h[2],0,h[3],H], fill=(255,0,0,70))
    
    crack_ov = np.zeros((H, W, 4), dtype=np.uint8)
    for _, r in df_comp.iterrows():
        crack_ov[labels == r["id"]] = COLOR_MAP.get(r["clase"], (0,0,0,160))
    
    out = Image.alpha_composite(base, zone_ov)
    out = Image.alpha_composite(out, Image.fromarray(crack_ov))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()

    for _, r in df_comp.iterrows():
        draw.text((r['x'], r['y']), r['clase'], fill=(0,0,0,255), font=font)
    return out.convert("RGB")

# --- APP ---
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🚀 Analizador de Pavimentos (Simetría Corregida)")
up = st.file_uploader("Subir imagen", type=["jpg", "png", "jpeg"])

if up and st.button("PROCESAR"):
    with st.spinner('Analizando...'):
        img_raw = Image.open(up).convert("RGB")
        h_obj = int(W_OBJETIVO * (img_raw.size[1] / img_raw.size[0]))
        img = img_raw.resize((W_OBJETIVO, h_obj), Image.Resampling.LANCZOS)
        
        arr = np.array(img)
        mask = adaptive_mask(arr)
        labels, ncomp = ndi.label(mask)
        slices = ndi.find_objects(labels)
        
        # Mapa de zonas para el resumen estadístico
        z_map = np.full(arr.shape[:2], -1)
        bounds, hp2, hp1 = compute_zone_bounds(W_OBJETIVO)
        z_map[:, bounds[1]:bounds[2]], z_map[:, bounds[2]:bounds[3]] = 1, 2
        for h in [hp2, hp1]:
            z_map[:, h[0]:h[1]], z_map[:, h[2]:h[3]] = 4, 4

        rows, prorr = [], []
        for i, slc in enumerate(slices, start=1):
            if slc is None or np.sum(labels[slc]==i) < 40: continue
            ys, xs = np.nonzero(labels[slc]==i)
            clase, Lpx, zona, xc, yc = classify_component(xs+slc[1].start, ys+slc[0].start, W_OBJETIVO)
            
            f = S_TRANSVERSAL if clase == "Transverse" else (S_LONGITUDINAL if clase in ["Longitudinal", "On Axis"] else S_TODAS)
            Lm = Lpx * f
            
            rows.append({"id":i, "clase":clase, "metros":Lm, "x":xc, "y":yc})
            
            # Determinar si es Wheel Path
            mask_c = (labels == i)
            is_wp = (np.sum(z_map[mask_c] == 4) / np.sum(mask_c)) >= HUELLA_MIN_FRAC
            prorr.append({"Zona": "Wheel Path" if is_wp else zona, "Tipo": clase, "Metros": Lm})

        df_p = pd.DataFrame(prorr).groupby(["Zona", "Tipo"])["Metros"].sum().reset_index() if prorr else pd.DataFrame()

        st.session_state.data = {
            "orig": img, 
            "proc": annotate_image_final(img, labels, pd.DataFrame(rows)), 
            "res": df_p
        }

if st.session_state.data:
    st.image(st.session_state.data["proc"], use_container_width=True)
    st.dataframe(st.session_state.data["res"], use_container_width=True)
