import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io

# ====================================================
# 1. FUNCIONES BASE
# ====================================================
def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    img_norm = img.astype(np.float64) / np.max(img)
    R, G, B = img_norm[:, :, 0], img_norm[:, :, 1], img_norm[:, :, 2]
    epsilon = 1e-10

    C_max = np.maximum(np.maximum(R, G), B)
    C_min = np.minimum(np.minimum(R, G), B)
    delta = C_max - C_min

    V = C_max
    S = np.where(C_max > epsilon, delta / (V + epsilon), 0.0)
    H = np.zeros_like(V)

    mask_r = (C_max == R) * (delta > epsilon)
    mask_g = (C_max == G) * (delta > epsilon)
    mask_b = (C_max == B) * (delta > epsilon)

    H[mask_r] = 60.0 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60.0 * (((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2)
    H[mask_b] = 60.0 * (((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4)

    H = np.where(H < 0, H + 360.0, H)
    H = np.where(delta < epsilon, 0.0, H)
    return np.stack([H, S, V], axis=-1)

def hsv_to_rgb(hsv_img: np.ndarray) -> np.ndarray:
    H, S, V = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    C = V * S
    X = C * (1.0 - np.abs(((H / 60.0) % 2) - 1.0))
    m = V - C

    R_prime = np.zeros_like(H)
    G_prime = np.zeros_like(H)
    B_prime = np.zeros_like(H)

    idx0 = (H >= 0) & (H < 60)
    idx1 = (H >= 60) & (H < 120)
    idx2 = (H >= 120) & (H < 180)
    idx3 = (H >= 180) & (H < 240)
    idx4 = (H >= 240) & (H < 300)
    idx5 = (H >= 300) & (H <= 360)

    R_prime[idx0], G_prime[idx0], B_prime[idx0] = C[idx0], X[idx0], 0
    R_prime[idx1], G_prime[idx1], B_prime[idx1] = X[idx1], C[idx1], 0
    R_prime[idx2], G_prime[idx2], B_prime[idx2] = 0, C[idx2], X[idx2]
    R_prime[idx3], G_prime[idx3], B_prime[idx3] = 0, X[idx3], C[idx3]
    R_prime[idx4], G_prime[idx4], B_prime[idx4] = X[idx4], 0, C[idx4]
    R_prime[idx5], G_prime[idx5], B_prime[idx5] = C[idx5], 0, X[idx5]

    return np.clip(np.stack([R_prime + m, G_prime + m, B_prime + m], axis=-1), 0.0, 1.0)

def funcion_gm(componente, factor_m, es_lch=False):
    resultado = componente * factor_m
    return np.maximum(0.0, resultado) if es_lch else np.clip(resultado, 0.0, 1.0)

def interpolar_factores_m(tonos_imagen, puntos_control):
    tonos = np.array([p[0] for p in puntos_control])
    factores = np.array([p[1] for p in puntos_control])
    tonos_ext = np.concatenate([tonos - 360.0, tonos, tonos + 360.0])
    factores_ext = np.concatenate([factores, factores, factores])
    orden = np.argsort(tonos_ext)
    return np.interp(tonos_imagen, tonos_ext[orden], factores_ext[orden])

def color_saturation(img_rgb, puntos_control, modo='HS'):
    img_norm = img_rgb.astype(np.float64) / np.max(img_rgb)
    if modo == 'HS':
        hsv = rgb_to_hsv(img_norm)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        m_pixel = interpolar_factores_m(H, puntos_control)
        S_nueva = funcion_gm(S, m_pixel, es_lch=False)
        return hsv_to_rgb(np.stack([H, S_nueva, V], axis=-1))
    elif modo == 'LCh':
        lab = color.rgb2lab(img_norm)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        croma = np.sqrt(a**2 + b**2)
        tono = np.degrees(np.arctan2(b, a)) % 360.0
        m_pixel = interpolar_factores_m(tono, puntos_control)
        croma_nuevo = funcion_gm(croma, m_pixel, es_lch=True)
        tono_rad = np.radians(tono)
        lab_mod = np.stack([L, croma_nuevo * np.cos(tono_rad), croma_nuevo * np.sin(tono_rad)], axis=-1)
        return np.clip(color.lab2rgb(lab_mod), 0.0, 1.0)

# ====================================================
# 2. DEFINICIÓN DE LAS 3 CONFIGURACIONES PREDICHO POR LA PAUTA
# ====================================================
configuraciones = [
    {
        "nombre": "1. Amplificación Selectiva (Tonos Cálidos)",
        "puntos": [(0, 1.0), (40, 2.5), (80, 2.5), (120, 1.0), (360, 1.0)],
        "color_plot": "orange"
    },
    {
        "nombre": "2. Atenuación Selectiva (Tonos Fríos)",
        "puntos": [(0, 1.0), (160, 1.0), (200, 0.05), (260, 0.05), (300, 1.0), (360, 1.0)],
        "color_plot": "blue"
    },
    {
        "nombre": "3. Combinada (Aumento y Disminución)",
        "puntos": [(0, 0.2), (60, 2.2), (120, 2.2), (220, 0.1), (360, 0.2)],
        "color_plot": "green"
    }
]

# ====================================================
# 3. GENERACIÓN DE LAS FIGURAS
# ====================================================
def ejecutar_experimento_completo(archivo_imagen):
    img_raw = io.imread(archivo_imagen)
    img_norm = img_raw.astype(np.float64) / np.max(img_raw)
    tonos_continuos = np.linspace(0, 360, 361)

    # FIGURA A: Las 3 Curvas m(h) juntas
    fig_curvas, axes_curvas = plt.subplots(1, 3, figsize=(18, 4))
    
    for idx, cfg in enumerate(configuraciones):
        pts_h = [p[0] for p in cfg["puntos"]]
        pts_m = [p[1] for p in cfg["puntos"]]
        m_interp = interpolar_factores_m(tonos_continuos, cfg["puntos"])

        axes_curvas[idx].plot(tonos_continuos, m_interp, color=cfg["color_plot"], linewidth=2.5, label='Curva $m(h)$')
        axes_curvas[idx].scatter(pts_h, pts_m, color='red', s=60, zorder=5, label='Puntos de Control')
        axes_curvas[idx].axhline(1.0, color='gray', linestyle='--', alpha=0.7, label='Neutro ($m=1$)')
        axes_curvas[idx].set_title(cfg["nombre"], fontsize=10)
        axes_curvas[idx].set_xlabel('Tono $h$ (grados)')
        axes_curvas[idx].set_ylabel('Factor $m$')
        axes_curvas[idx].set_xlim(0, 360)
        axes_curvas[idx].grid(True, linestyle=':', alpha=0.6)
        axes_curvas[idx].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('grafico_3_curvas_m.png', dpi=300)
    print("-> Gráfico de las 3 curvas m(h) guardado como 'grafico_3_curvas_m.png'.")
    plt.show()

    # FIGURA B: Resultados visuales (Original vs HS vs LCh para las 3 configuraciones)
    fig_img, axes_img = plt.subplots(3, 3, figsize=(15, 12))

    for row, cfg in enumerate(configuraciones):
        res_hs = color_saturation(img_norm, cfg["puntos"], modo='HS')
        res_lch = color_saturation(img_norm, cfg["puntos"], modo='LCh')

        axes_img[row, 0].imshow(img_norm)
        axes_img[row, 0].set_title("Original")
        axes_img[row, 0].axis('off')

        axes_img[row, 1].imshow(res_hs)
        axes_img[row, 1].set_title(f"Modo HS\n{cfg['nombre']}")
        axes_img[row, 1].axis('off')

        axes_img[row, 2].imshow(res_lch)
        axes_img[row, 2].set_title(f"Modo LCh\n{cfg['nombre']}")
        axes_img[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig('resultados_visuales_3_configuraciones.png', dpi=300)
    print("-> Comparativa de imágenes guardada como 'resultados_visuales_3_configuraciones.png'.")
    plt.show()

if __name__ == '__main__':
    nombre_imagen = 'P1_IMG_2402.tif'  # Reemplaza por tu archivo de imagen
    ejecutar_experimento_completo(nombre_imagen)