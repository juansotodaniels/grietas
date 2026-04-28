import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.colors as mcolors
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Analizador MEL - Digitalización")

# --- CONSTANTES TÉCNICAS (Tu original) ---
ANCHO_BERMA, ANCHO_PISTA = 2.0, 3.5
ANCHO_TOTAL = 11.0 # 2+3.5+3.5+2
ANG_TOL, EJE_TOL_M = 20.0, 0.2
MIN_YELLOW_PIXELS, MIN_YELLOW_RATIO = 150, 0.0005
S_MIN_COLOR, V_MIN_COLOR = 0.25, 0.15
BLUE_H_MIN, BLUE_H_MAX = 200.0/360.0, 260.0/360.0
S_TRANSVERSAL, S_LONGITUDINAL, S_TODAS = 0.005368, 0.13644, 0.13655
CENTER_OFFSET, PINK_LEFT_OFFSET, PINK_RIGHT_OFFSET = 220, 340, 620
HUELLA_MIN_FRAC = 0.75

COLOR_MAP = {
    "longitudinal": (255, 0, 0, 160),
    "transversal": (0, 102, 255, 160),
    "en_el_eje": (0, 180, 0, 180),
    "en_todas_direcciones": (200, 0, 200, 160)
}

# --- FUNCIONES CORE ---

def yellow_mask_rgb_hsv(arr):
    rgb = arr.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(rgb)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (V >= V_MIN_COLOR) & (S >= S_MIN_COLOR) & ~((H >= BLUE_H_MIN) & (H <= BLUE_H_MAX))
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)))
    return ndi.binary_opening(mask, structure=np.ones((2, 2)))

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
    if cov.ndim < 2: return "desconocido", 0, "Fuera", x_c, y_c
    evals, evecs = np.linalg.eig(cov)
    major = evecs[:, np.argmax(evals)]
    ang = abs(np.degrees(np.arctan2(major[1], major[0]))) % 180.0
    
    (x0, x1, x2, x3, x4), _, _ = compute_zone_bounds(W)
    zona = "Berma Izq" if x_c < x1 else "P2" if x_c < x2 else "P1" if x_c < x3 else "Berma Der"
    
    x_p2_eje = x1 + (ANCHO_BERMA + ANCHO_PISTA) * ((x3-x1)/ANCHO_TOTAL)
    dist_min_m = float(np.min(np.abs(xs - x_p2_eje))) * S_LONGITUDINAL

    if abs(ang - 90.0) <= ANG_TOL:
        clase = "en_el_eje" if dist_min_m <= EJE_TOL_M else "longitudinal"
    elif (ang <= ANG_TOL) or (ang >= 180.0 - ANG_TOL):
        clase = "transversal"
    else:
        clase = "en_todas_direcciones"
    
    Lpx = float(np.ptp(major[0]*(xs-x_c) + major[1]*(ys-y_c)))
    return clase, Lpx, zona, x_c, y_c

def annotate_image_final(img_pil, labels, df_comp):
    base = img_pil.convert("RGBA")
    W, H = base.size
    zone_ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    zdraw = ImageDraw.Draw(zone_ov)
    (x0, x1, x2, x3, x4), hp2, hp1 = compute_zone_bounds(W)
    
    cols = {"BI": (255,255,0,70), "P2": (0,200,255,70), "P1": (255,150,255,70), "BD": (150,230,0,70), "H": (255,0,0,100)}
    if x1>x0: zdraw.rectangle([x0,0,x1,H], fill=cols["BI"])
    if x2>x1: zdraw.rectangle([x1,0,x2,H], fill=cols["P2"])
    if x3>x2: zdraw.rectangle([x2,0,x3,H], fill=cols["P1"])
    if x4>x3: zdraw.rectangle([x3,0,x4,H], fill=cols["BD"])
    # Huellas
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
        txt = f"{r['clase'][0].upper()}: {r['metros']:.2f}m"
        draw.rectangle([r['x']-2, r['y']-2, r['x']+80, r['y']+14], fill=(255,255,255,200))
        draw.text((r['x'], r['y']), txt, fill=(0,0,0,255), font=font)
    return out.convert("RGB")

# --- APP ---
if 'data' not in st.session_state:
    st.session_state.data = None

st.sidebar.title("Configuración")
pk_ini = st.sidebar.number_input("PK Inicio", value=119.500, format="%.3f")
pk_fin = st.sidebar.number_input("PK Fin", value=119.750, format="%.3f")
mode = st.sidebar.radio("Pantalla", ["Carga iPad", "Monitor Resultados"])

if mode == "Carga iPad":
    st.title("📤 Entrada de Terreno")
    up = st.file_uploader("Captura/Imagen", type=["jpg", "png", "jpeg"])
    if up and st.button("PROCESAR"):
        img = Image.open(up).convert("RGB")
        arr = np.array(img)
        mask = yellow_mask_rgb_hsv(arr)
        labels, ncomp = ndi.label(mask)
        slices = ndi.find_objects(labels)
        
        # Build Zone Map for Prorate
        z_map = np.full(arr.shape[:2], -1)
        (x0,x1,x2,x3,x4), hp2, hp1 = compute_zone_bounds(arr.shape[1])
        z_map[:, x0:x1], z_map[:, x1:x2], z_map[:, x2:x3], z_map[:, x3:x4] = 0, 1, 2, 3
        for h in [hp2, hp1]:
            if h[1]>h[0]: z_map[:, h[0]:h[1]] = 4
            if h[3]>h[2]: z_map[:, h[2]:h[3]] = 4

        rows, prorr = [], []
        for i, slc in enumerate(slices, start=1):
            if slc is None or np.sum(labels[slc]==i) < 80: continue
            ys, xs = np.nonzero(labels[slc]==i)
            clase, Lpx, zona, xc, yc = classify_component(xs+slc[1].start, ys+slc[0].start, arr.shape[1])
            f = S_TRANSVERSAL if clase=="transversal" else S_LONGITUDINAL if clase in ["longitudinal","en_el_eje"] else S_TODAS
            Lm = Lpx * f
            
            rows.append({"id":i, "clase":clase, "metros":Lm, "x":xc, "y":yc})
            # Prorrateo
            mask_c = (labels == i)
            ratio_h = np.sum(z_map[mask_c] == 4) / np.sum(mask_c)
            z_final = "Huella" if ratio_h >= HUELLA_MIN_FRAC else zona
            prorr.append({"Zona": z_final, "Clase": clase, "Metros": Lm})

        df_p = pd.DataFrame(prorr).groupby(["Zona", "Clase"])["Metros"].sum().reset_index()
        st.session_state.data = {"orig": img, "proc": annotate_image_final(img, labels, pd.DataFrame(rows)), "res": df_p}
        st.success("Procesamiento completo.")

else:
    st.title("📊 Monitor de Inspección")
    if st.session_state.data:
        c1, c2 = st.columns(2)
        c1.image(st.session_state.data["orig"], use_container_width=True, caption=f"Original PK {pk_ini}")
        c2.image(st.session_state.data["proc"], use_container_width=True, caption="Análisis de Pistas y Huellas")
        st.subheader("Resumen de Hallazgos")
        st.dataframe(st.session_state.data["res"], use_container_width=True)
        if st.button("Limpiar"):
            st.session_state.data = None
            st.rerun()
    else:
        st.info("Esperando datos...")
