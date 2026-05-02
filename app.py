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
CENTER_OFFSET = 240         
ANCHO_PISTA_PX = 467        

PINK_LEFT_OFFSET = ANCHO_PISTA_PX  
PINK_RIGHT_OFFSET = ANCHO_PISTA_PX

WP_OFFSET = 110             
WP_WIDTH = 95               

ANG_TOL = 20.0
EJE_TOL_M = 0.05            
HUELLA_MIN_FRAC = 0.75

S_LONGITUDINAL = 0.075196
S_TRANSVERSAL = 0.007338
S_TODAS = 0.036909

# Definición de colores para el pintado de grietas
COLOR_MAP = {
    "Longitudinal": (255, 0, 0, 255),        # Rojo
    "Transverse": (0, 102, 255, 255),      # Azul
    "On Axis": (0, 180, 0, 255),           # Verde
    "In all directions": (200, 0, 200, 255) # Púrpura
}

# --- FUNCIONES DE APOYO ---
def get_image_download_link(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def adaptive_crack_mask(arr):
    # Detectamos contenido que no sea el fondo (asumiendo fondo claro)
    gray = np.mean(arr, axis=2)
    mask = gray < 220 
    # Limpieza suave para no borrar líneas finas
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))
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
    
    # Manejo de casos con muy pocos píxeles
    if Xpx.shape[1] < 5: return "In all directions", 0, "Outside", x_c, y_c
    
    cov = np.cov(Xpx)
    if cov.ndim < 2 or np.isnan(cov).any(): 
        return "In all directions", 0, "Outside", x_c, y_c
        
    evals, evecs = np.linalg.eig(cov)
    idx_max = np.argmax(evals)
    major_val = evals[idx_max]
    minor_val = evals[1 - idx_max]
    
    # Relación de aspecto para detectar "In all directions" (púrpura)
    # Si la forma es muy redondeada o ramificada, la covarianza es similar en ambos ejes
    if major_val > 0 and (minor_val / major_val) > 0.4:
        clase_base = "In all directions"
    else:
        major = evecs[:, idx_max]
        ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
        
        if abs(ang - 90.0) <= ANG_TOL:
            clase_base = "Longitudinal"
        elif (ang <= ANG_TOL) or (ang >= 180.0 - ANG_TOL):
            clase_base = "Transverse"
        else:
            clase_base = "In all directions"
    
    # Determinación de zona
    (_, x1, x2, x3, _), _, _ = compute_zone_bounds(W)
    if x_c < x1: zona = "Left Shoulder"
    elif x_c < x2: zona = "Lane 2"
    elif x_c < x3: zona = "Lane 1"
    else: zona = "Right Shoulder"
    
    # Especial para "On Axis" (Eje central)
    if clase_base == "Longitudinal" and abs(x_c - x2) * S_LONGITUDINAL <= EJE_TOL_M:
        clase_base = "On Axis"

    Lpx = float(np.ptp(Xpx[0]) + np.ptp(Xpx[1])) # Estimación de longitud
    return clase_base, Lpx, zona, x_c, y_c

def annotate_image_final(img_pil, labels, df_comp):
    base = img_pil.convert("RGBA")
    W, H = base.size
    
    # 1. Capa de Zonas (Fondo)
    zone_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    zdraw = ImageDraw.Draw(zone_ov)
    (x0, x1, x2, x3, x4), hp2, hp1 = compute_zone_bounds(W)
    
    zdraw.rectangle([x0, 0, x1, H], fill=(255,255,0,40)) # Berma
    zdraw.rectangle([x1, 0, x2, H], fill=(0,200,255,40)) # Lane 2
    zdraw.rectangle([x2, 0, x3, H], fill=(255,150,255,40))# Lane 1
    zdraw.rectangle([x3, 0, x4, H], fill=(255,255,0,40)) # Berma
    
    # 2. Capa de Huellas (Rojo suave)
    for h in [hp2, hp1]:
        zdraw.rectangle([h[0], 0, h[1], H], fill=(255, 0, 0, 60))
        zdraw.rectangle([h[2], 0, h[3], H], fill=(255, 0, 0, 60))

    # 3. Capa de Grietas (Colores específicos)
    crack_ov = np.zeros((H, W, 4), dtype=np.uint8)
    for _, r in df_comp.iterrows():
        color = COLOR_MAP.get(r["clase"], (128, 128, 128, 255))
        crack_ov[labels == r["id"]] = color
    
    # Combinar capas
    out = Image.alpha_composite(base, zone_ov)
    out = Image.alpha_composite(out, Image.fromarray(crack_ov))
    
    # 4. Etiquetas de texto
    draw = ImageDraw.Draw(out)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 22) 
    except: font = ImageFont.load_default()

    for _, r in df_comp.iterrows():
        txt = f"{r['clase']}"
        draw.text((r['x']+5, r['y']), txt, fill=(0,0,0,255), font=font)
        
    return out.convert("RGB")

# --- APP ---
if 'data' not in st.session_state:
    st.session_state.data = None

st.sidebar.title("Navegación")
mode = st.sidebar.radio("Ir a:", ["Carga de Datos", "Monitor de Resultados"])

if mode == "Carga de Datos":
    st.title("🚀 Analizador de Grietas")
    up = st.file_uploader("Subir imagen de pavimento", type=["jpg", "png", "jpeg"])
    
    if up and st.button("PROCESAR"):
        with st.spinner('Analizando patrones...'):
            img_raw = Image.open(up).convert("RGB")
            img = img_raw.resize((W_OBJETIVO, int(W_OBJETIVO*(img_raw.size[1]/img_raw.size[0]))), Image.Resampling.LANCZOS)
            
            arr = np.array(img)
            mask = adaptive_crack_mask(arr)
            labels, ncomp = ndi.label(mask)
            slices = ndi.find_objects(labels)
            
            z_map = np.full(arr.shape[:2], -1)
            (x0,x1,x2,x3,x4), hp2, hp1 = compute_zone_bounds(W_OBJETIVO)
            z_map[:, x1:x2], z_map[:, x2:x3] = 1, 2
            for h in [hp2, hp1]:
                z_map[:, h[0]:h[1]] = 4
                z_map[:, h[2]:h[3]] = 4

            rows, prorr = [], []
            for i, slc in enumerate(slices, start=1):
                if slc is None or np.sum(labels[slc]==i) < 40: continue 
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
                "proc": annotate_image_final(img, labels, pd.DataFrame(rows)), 
                "res": df_p
            }
            st.success(f"Análisis completo: {len(rows)} grietas clasificadas.")

else:
    st.title("📊 Monitor de Resultados")
    if st.session_state.data:
        st.image(st.session_state.data["proc"], use_container_width=True)
        st.divider()
        st.subheader("Resumen de Metros Lineales por Zona")
        st.table(st.session_state.data["res"])
