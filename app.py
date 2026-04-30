import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.colors as mcolors

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Diagnóstico de Coordenadas - Pavimentos")

# --- CONSTANTES TÉCNICAS (Calibradas para 2550px) ---
W_OBJETIVO = 2550
CENTER_OFFSET = 240 # Para llegar a 1515: (2550/2) + 240
PINK_LEFT_OFFSET, PINK_RIGHT_OFFSET = 340, 620
EJE_TOL_M = 0.05 # Tolerancia estricta de 5cm para evitar falsos "On Axis"
S_LONGITUDINAL = 0.075196

# --- FUNCIONES ---
def compute_zone_bounds(W):
    x_center_shifted = (W / 2.0) + CENTER_OFFSET
    x1 = int(round(max(0, min(W, x_center_shifted - PINK_LEFT_OFFSET))))
    x3 = int(round(max(0, min(W, x_center_shifted + PINK_RIGHT_OFFSET))))
    x2 = int(round(x_center_shifted)) 
    return (0, x1, x2, x3, W)

def yellow_mask_rgb_hsv(arr):
    rgb = arr.astype(np.float32) / 255.0
    hsv = mcolors.rgb_to_hsv(rgb)
    mask = (hsv[..., 2] < 0.35) | ((hsv[..., 1] > 0.25) & (hsv[..., 2] > 0.40))
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    return mask

# --- APP ---
st.title("🛠️ Monitor de Diagnóstico de Coordenadas")

up = st.file_uploader("Subir imagen de prueba", type=["jpg", "png", "jpeg"])

if up:
    img_raw = Image.open(up).convert("RGB")
    aspect = img_raw.size[1] / img_raw.size[0]
    h_obj = int(W_OBJETIVO * aspect)
    img = img_raw.resize((W_OBJETIVO, h_obj), Image.Resampling.LANCZOS)
    
    # 1. CÁLCULO DE COORDENADAS TEÓRICAS
    x0, x1, x2, x3, x4 = compute_zone_bounds(W_OBJETIVO)
    
    st.subheader("1. Informe de Calibración (Píxeles)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ancho Objetivo ($W$)", W_OBJETIVO)
        st.metric("Centro Real ($W/2$)", W_OBJETIVO/2)
    with col2:
        st.metric("OFFSET Manual", CENTER_OFFSET)
        st.metric("Eje Calculado ($x2$)", x2)
    with col3:
        st.write("**Límites de Zonas:**")
        st.write(f"- Berma Izq: {x0} a {x1}")
        st.write(f"- Carril 2: {x1} a {x2}")
        st.write(f"- Carril 1: {x2} a {x3}")
        st.write(f"- Berma Der: {x3} a {x4}")

    # 2. PROCESAMIENTO Y DETECCIÓN
    arr = np.array(img)
    mask = yellow_mask_rgb_hsv(arr)
    labels, ncomp = ndi.label(mask)
    slices = ndi.find_objects(labels)
    
    debug_data = []
    for i, slc in enumerate(slices, start=1):
        if slc is None or np.sum(labels[slc]==i) < 250: continue
        ys, xs = np.nonzero(labels[slc]==i)
        x_abs = xs + slc[1].start
        y_abs = ys + slc[0].start
        x_centro_crack = x_abs.mean()
        
        # Lógica de clasificación
        dist_al_eje_px = np.min(np.abs(x_abs - x2))
        dist_m = float(dist_al_eje_px) * S_LONGITUDINAL
        es_on_axis = dist_m <= EJE_TOL_M
        
        debug_data.append({
            "ID": i,
            "X_Centro_Px": round(x_centro_crack, 1),
            "Dist_al_Eje_Px": round(dist_al_eje_px, 1),
            "Dist_Metros": round(dist_m, 3),
            "Es_On_Axis": "SÍ" if es_on_axis else "NO"
        })

    # 3. VISUALIZACIÓN DE RESULTADOS
    st.divider()
    st.subheader("2. Análisis de Objetos Detectados")
    if debug_data:
        df_debug = pd.DataFrame(debug_data)
        st.table(df_debug)
        
        # Dibujar líneas de diagnóstico sobre la imagen
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        # Dibujamos el eje X2 en AZUL para verificar dónde lo pone el código
        draw.line([(x2, 0), (x2, h_obj)], fill="blue", width=5)
        # Dibujamos los límites X1 y X3 en AMARILLO
        draw.line([(x1, 0), (x1, h_obj)], fill="yellow", width=3)
        draw.line([(x3, 0), (x3, h_obj)], fill="yellow", width=3)
        
        st.image(draw_img, caption="Línea AZUL = Eje Real del Código | Líneas AMARILLAS = Límites de Pistas")
    else:
        st.warning("No se detectaron grietas para analizar.")
