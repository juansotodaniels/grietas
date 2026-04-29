import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.colors as mcolors
import io
import base64

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Analizador de Pavimentos - Filtro Base")

# --- CONSTANTES TÉCNICAS ---
ANCHO_BERMA, ANCHO_PISTA = 2.0, 3.5
ANCHO_TOTAL = 11.0 
ANG_TOL, EJE_TOL_M = 20.0, 0.2
MIN_YELLOW_PIXELS, MIN_YELLOW_RATIO = 150, 0.0005
BLUE_H_MIN, BLUE_H_MAX = 200.0/360.0, 260.0/360.0

# ESCALAS CALIBRADAS
S_LONGITUDINAL = 0.075196
S_TRANSVERSAL = 0.007338
S_TODAS = 0.036909

CENTER_OFFSET, PINK_LEFT_OFFSET, PINK_RIGHT_OFFSET = 180, 340, 620
HUELLA_MIN_FRAC = 0.75

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

# --- MOTOR DE PROCESAMIENTO CON FILTRO DE TEXTOS Y LÍNEAS GRISES ---
def yellow_mask_rgb_hsv(arr):
    rgb = arr.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(rgb)
    S = hsv[..., 1]
    V = hsv[..., 2]
    
    # 1. Detección de trazos (Negro y Colores)
    mask_black = (V < 0.35)
    mask_colors = (S > 0.25) & (V > 0.40)
    mask = mask_black | mask_colors
    
    # 2. FILTRO CRÍTICO: Eliminación de líneas finas y textos
    # Usamos una apertura (opening) más agresiva para eliminar elementos delgados
    # como las líneas grises de la cuadrícula y las letras de la plantilla.
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    
    # 3. Re-unión de trazos de grietas legítimas
    mask = ndi.binary_closing(mask, structure=np.ones((4, 4)))
    
    return mask

def compute_zone_bounds(W):
    x_center_shifted = (W / 2.0) + CENTER_OFFSET
    x1 = int(round(max(0, min(W, x_center_shifted - PINK_LEFT_OFFSET))))
    x3 = int(round(max(0, min(W, x_center_shifted + PINK_RIGHT_OFFSET))))
    x2 = int(round((x1 + x3) / 2.0))
    def get_h(center_p, off, width, x_min, x_max):
        c = center_p + off
        return int(round(max(x_min, min(x_max, c - width/2.0)))), int(round(max(x_min, min(x_max, c + width/2.0))))
    h_p2 = (*get_h((x1+x2)/2.0, -100, 100, x1, x2), *get_h((x1+x2)/2.0, 100, 85, x1, x2))
    h_p1 = (*get_h((x2+x3)/2.0, -120, 100, x2, x3), *get_h((x2+x3)/2.0, 120, 100, x2, x3))
    return (0, x1, x2, x3, W), h_p2, h_p1

def classify_component(xs, ys, W):
    x_c, y_c = xs.mean(), ys.mean()
    Xpx = np.vstack([xs - x_c, ys - y_c])
    cov = np.cov(Xpx)
    if cov.ndim < 2: return "Unknown", 0, "Outside", x_c, y_c
    evals, evecs = np.linalg.eig(cov)
    major = evecs[:, np.argmax(evals)]
    ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
    
    (x0, x1, x2, x3, x4), _, _ = compute_zone_bounds(W)
    
    if x_c < x1: zona = "Left Shoulder"
    elif x_c < x2: zona = "Lane 2"
    elif x_c < x3: zona = "Lane 1"
    else: zona = "Right Shoulder"
    
    x_p2_eje = x1 + (ANCHO_BERMA + ANCHO_PISTA) * ((x3-x1)/ANCHO_TOTAL)
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
    (x0, x1, x2, x3, x4), hp2, hp1 = compute_zone_bounds(W)
    
    cols = {"SHOULDER": (255,255,0,70), "LANE2": (0,200,255,70), "LANE1": (255,150,255,70)}
    if x1>x0: zdraw.rectangle([x0,0,x1,H], fill=cols["SHOULDER"])
    if x2>x1: zdraw.rectangle([x1,0,x2,H], fill=cols["LANE2"])
    if x3>x2: zdraw.rectangle([x2,0,x3,H], fill=cols["LANE1"])
    if x4>x3: zdraw.rectangle([x3,0,x4,H], fill=cols["SHOULDER"])
    
    for h in [hp2, hp1]:
        if h[1]>h[0]: zdraw.rectangle([h[0],0,h[1],H], fill=(255,0,0,100))
        if h[3]>h[2]: zdraw.rectangle([h[2],0,h[3],H], fill=(255,0,0,100))

    crack_ov = np.zeros((H, W, 4), dtype=np.uint8)
    for _, r in df_comp.iterrows():
        crack_ov[labels == r["id"]] = COLOR_MAP.get(r["clase"], (0,0,0,160))
    
    out = Image.alpha_composite(base, zone_ov)
    out = Image.alpha_composite(out, Image.fromarray(crack_ov))
    draw = ImageDraw.Draw(out)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except: font = ImageFont.load_default()

    for _, r in df_comp.iterrows():
        txt = f"{r['clase']}: {r['metros']:.2f}m"
        draw.rectangle([r['x']-2, r['y']-2, r['x']+110, r['y']+14], fill=(255,255,255,200))
        draw.text((r['x'], r['y']), txt, fill=(0,0,0,255), font=font)
    return out.convert("RGB")

# --- APP ---
if 'data' not in st.session_state:
    st.session_state.data = None

st.sidebar.title("Navigation")
mode = st.sidebar.radio("Go to:", ["Field Upload", "Results Monitor"])

if mode == "Field Upload":
    st.title("📤 Road Data Input")
    up = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if up and st.button("🚀 PROCESS IMAGE"):
        with st.spinner('Analyzing infrastructure...'):
            img = Image.open(up).convert("RGB")
            arr = np.array(img)
            mask = yellow_mask_rgb_hsv(arr)
            labels, ncomp = ndi.label(mask)
            slices = ndi.find_objects(labels)
            
            z_map = np.full(arr.shape[:2], -1)
            (x0,x1,x2,x3,x4), hp2, hp1 = compute_zone_bounds(arr.shape[1])
            z_map[:, x0:x1], z_map[:, x1:x2], z_map[:, x2:x3], z_map[:, x3:x4] = 0, 1, 2, 3
            for h in [hp2, hp1]:
                if h[1]>h[0]: z_map[:, h[0]:h[1]] = 4
                if h[3]>h[2]: z_map[:, h[2]:h[3]] = 4

            rows, prorr = [], []
            for i, slc in enumerate(slices, start=1):
                # Filtro de área mínimo aumentado para asegurar que textos pequeños no pasen
                if slc is None or np.sum(labels[slc]==i) < 200: continue
                ys, xs = np.nonzero(labels[slc]==i)
                clase, Lpx, zona, xc, yc = classify_component(xs+slc[1].start, ys+slc[0].start, arr.shape[1])
                
                if clase == "Transverse": f = S_TRANSVERSAL
                elif clase in ["Longitudinal", "On Axis"]: f = S_LONGITUDINAL
                else: f = S_TODAS
                
                Lm = Lpx * f
                rows.append({"id":i, "clase":clase, "metros":Lm, "x":xc, "y":yc})
                mask_c = (labels == i)
                ratio_h = np.sum(z_map[mask_c] == 4) / np.sum(mask_c)
                z_final = "Wheel Path" if ratio_h >= HUELLA_MIN_FRAC else zona
                prorr.append({"Zone": z_final, "Type": clase, "Meters": Lm})

            if len(prorr) > 0:
                df_p = pd.DataFrame(prorr).groupby(["Zone", "Type"])["Meters"].sum().reset_index()
            else:
                df_p = pd.DataFrame(columns=["Zone", "Type", "Meters"])

            st.session_state.data = {
                "orig": img, 
                "proc": annotate_image_final(img, labels, pd.DataFrame(rows)) if len(rows) > 0 else img, 
                "res": df_p
            }
            if len(prorr) == 0: st.warning("No distress detected.")
            else: st.success("Processing complete. Template lines and text ignored.")

else:
    # (Resto del código de visualización idéntico)
    st.title("📊 Analysis Results")
    if st.session_state.data:
        st.info("💡 **TIP:** Hover over the right image to zoom.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            st.image(st.session_state.data["orig"], use_container_width=True)
        with c2:
            st.subheader("Distress Map (Zoomed)")
            img_b64 = get_image_download_link(st.session_state.data["proc"])
            zoom_html = f"""
            <div style="overflow: hidden; border: 1px solid #444; border-radius: 10px;">
                <img src="{img_b64}" 
                     style="width: 100%; transition: transform .3s ease; cursor: crosshair;" 
                     onmouseover="this.style.transform='scale(2.5)'" 
                     onmouseout="this.style.transform='scale(1)'"
                     onmousemove="this.style.transformOrigin = ((event.offsetX / this.width) * 100) + '% ' + ((event.offsetY / this.height) * 100) + '%'">
            </div>
            """
            st.components.v1.html(zoom_html, height=600)
        
        st.divider()
        st.subheader("📋 Calculation Summary")
        st.dataframe(st.session_state.data["res"], use_container_width=True)
        
        if st.button("🗑️ CLEAR DATA"):
            st.session_state.data = None
            st.rerun()
