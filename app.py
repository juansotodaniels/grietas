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

# --- AJUSTE DE CONSTANTES ---
W_OBJETIVO = 2550
CENTER_OFFSET = 240         # Mantiene el eje en el píxel 1515
ANCHO_PISTA_PX = 467        # Ancho de las pistas en píxeles

PINK_LEFT_OFFSET = ANCHO_PISTA_PX  
PINK_RIGHT_OFFSET = ANCHO_PISTA_PX

WP_OFFSET = 110             # Distancia centro carril a huella
WP_WIDTH = 95               # Ancho de franja de huella

ANG_TOL = 20.0
EJE_TOL_M = 0.05            # Tolerancia 5cm
HUELLA_MIN_FRAC = 0.75

S_LONGITUDINAL = 0.075196
S_TRANSVERSAL = 0.007338
S_TODAS = 0.036909

COLOR_MAP = {
    "Longitudinal": (255, 0, 0, 160),
    "Transverse": (0, 102, 255, 160),
    "On Axis": (0, 180, 0, 180),
    "In all directions": (200, 0, 200, 160)
}

# --- FUNCIONES DE APOYO ---
def get_image_download_link(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def universal_mask(arr):
    # Nueva máscara optimizada para detectar trazos (negro/rojo) sobre fondos claros
    # Convierte a escala de grises para detectar contraste
    gray = np.mean(arr, axis=2)
    # Detecta lo que NO es blanco/crema (intensidad < 200)
    mask = gray < 200 
    # Limpieza de ruido
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    return mask

def compute_zone_bounds(W):
    x2 = int(round((W / 2.0) + CENTER_OFFSET))
    x1 = int(round(max(0, x2 - PINK_LEFT_OFFSET)))
    x3 = int(round(min(W, x2 + PINK_RIGHT_OFFSET)))
    
    def get_h(center_p, off, width, x_min, x_max):
        c = center_p + off
        return int(round(max(x_min, min(x_max, c - width/2.0)))), int(round(max(x_min, min(x_max, c + width/2.0))))
    
    c_lane2 = (x1 + x2) / 2.0
    c_lane1 = (x2 + x3) / 2.0
    
    h_p2 = (*get_h(c_lane2, -WP_OFFSET, WP_WIDTH, x1, x2), *get_h(c_lane2, WP_OFFSET, WP_WIDTH, x1, x2))
    h_p1 = (*get_h(c_lane1, -WP_OFFSET, WP_WIDTH, x2, x3), *get_h(c_lane1, WP_OFFSET, WP_WIDTH, x2, x3))
    
    return (0, x1, x2, x3, W), h_p2, h_p1

def classify_component(xs, ys, W):
    x_c, y_c = xs.mean(), ys.mean()
    Xpx = np.vstack([xs - x_c, ys - y_c])
    cov = np.cov(Xpx)
    if cov.ndim < 2: return "Unknown", 0, "Outside", x_c, y_c
    evals, evecs = np.linalg.eig(cov)
    major = evecs[:, np.argmax(evals)]
    ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
    
    (_, x1, x2, x3, _), _, _ = compute_zone_bounds(W)
    
    if x_c < x1: zona = "Left Shoulder"
    elif x_c < x2: zona = "Lane 2"
    elif x_c < x3: zona = "Lane 1"
    else: zona = "Right Shoulder"
    
    dist_al_eje_m = abs(x_c - x2) * S_LONGITUDINAL

    if abs(ang - 90.0) <= ANG_TOL:
        clase = "On Axis" if dist_al_eje_m <= EJE_TOL_M else "Longitudinal"
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
    (x0, x1, x2, x3, x4), hp2, hp1 = compute_zone_bounds(W)
    
    cols = {"SHOULDER": (255,255,0,50), "LANE2": (0,200,255,50), "LANE1": (255,150,255,50)}
    zdraw.rectangle([x0, 0, x1, H], fill=cols["SHOULDER"])
    zdraw.rectangle([x1, 0, x2, H], fill=cols["LANE2"])
    zdraw.rectangle([x2, 0, x3, H], fill=cols["LANE1"])
    zdraw.rectangle([x3, 0, x4, H], fill=cols["SHOULDER"])
    
    for h in [hp2, hp1]:
        if h[1] > h[0]: zdraw.rectangle([h[0], 0, h[1], H], fill=(255, 0, 0, 80))
        if h[3] > h[2]: zdraw.rectangle([h[2], 0, h[3], H], fill=(255, 0, 0, 80))

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
        draw.rectangle([r['x']-2, r['y']-2, r['x']+240, r['y']+30], fill=(255,255,255,180))
        draw.text((r['x'], r['y']), txt, fill=(0,0,0,255), font=font)
    return out.convert("RGB")

# --- APP ---
if 'data' not in st.session_state:
    st.session_state.data = None

st.sidebar.title("Navegación")
mode = st.sidebar.radio("Ir a:", ["Carga de Datos", "Monitor de Resultados"])

if mode == "Carga de Datos":
    st.title("📤 Analizador Vial")
    up = st.file_uploader("Subir imagen", type=["jpg", "png", "jpeg"])
    if up and st.button("🚀 PROCESAR IMAGEN"):
        with st.spinner('Analizando...'):
            img_raw = Image.open(up).convert("RGB")
            h_obj = int(W_OBJETIVO * (img_raw.size[1] / img_raw.size[0]))
            img = img_raw.resize((W_OBJETIVO, h_obj), Image.Resampling.LANCZOS)
            
            arr = np.array(img)
            mask = universal_mask(arr) # Cambio de máscara
            labels, ncomp = ndi.label(mask)
            slices = ndi.find_objects(labels)
            
            z_map = np.full(arr.shape[:2], -1)
            (x0,x1,x2,x3,x4), hp2, hp1 = compute_zone_bounds(W_OBJETIVO)
            z_map[:, x0:x1], z_map[:, x1:x2], z_map[:, x2:x3], z_map[:, x3:x4] = 0, 1, 2, 3
            for h in [hp2, hp1]:
                if h[1]>h[0]: z_map[:, h[0]:h[1]] = 4
                if h[3]>h[2]: z_map[:, h[2]:h[3]] = 4

            rows, prorr = [], []
            for i, slc in enumerate(slices, start=1):
                # Bajamos el umbral de píxeles para detectar trazos finos
                if slc is None or np.sum(labels[slc]==i) < 50: continue 
                ys, xs = np.nonzero(labels[slc]==i)
                clase, Lpx, zona, xc, yc = classify_component(xs+slc[1].start, ys+slc[0].start, W_OBJETIVO)
                
                f = S_TRANSVERSAL if clase == "Transverse" else (S_LONGITUDINAL if clase in ["Longitudinal", "On Axis"] else S_TODAS)
                Lm = Lpx * f
                rows.append({"id":i, "clase":clase, "metros":Lm, "x":xc, "y":yc})
                mask_c = (labels == i)
                ratio_h = np.sum(z_map[mask_c] == 4) / np.sum(mask_c)
                z_final = "Wheel Path" if ratio_h >= HUELLA_MIN_FRAC else zona
                prorr.append({"Zone": z_final, "Type": clase, "Meters": Lm})

            df_p = pd.DataFrame(prorr).groupby(["Zone", "Type"])["Meters"].sum().reset_index() if prorr else pd.DataFrame()

            st.session_state.data = {
                "orig": img, 
                "proc": annotate_image_final(img, labels, pd.DataFrame(rows)) if rows else img, 
                "res": df_p
            }
            st.success(f"Detecciones realizadas: {len(rows)}")

else:
    st.title("📊 Resultados")
    if st.session_state.data:
        c1, c2 = st.columns(2)
        with c1: st.image(st.session_state.data["orig"], caption="Original", use_container_width=True)
        with c2:
            img_b64 = get_image_download_link(st.session_state.data["proc"])
            st.components.v1.html(f'<img src="{img_b64}" style="width:100%;">', height=600)
        
        st.divider()
        st.dataframe(st.session_state.data["res"], use_container_width=True)
        if st.button("🗑️ LIMPIAR"):
            st.session_state.data = None
            st.rerun()
