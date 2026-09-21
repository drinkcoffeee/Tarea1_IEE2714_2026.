import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io

# ====================================================
# 1. CONVERSIÓN RGB <-> HSV (Manual)
# ====================================================
def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """Convierte RGB a HSV (H [0, 360], S [0, 1], V [0, 1])."""
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
    """Reconvierte de HSV (H en [0, 360]) a RGB [0, 1]."""
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

    rgb_img = np.stack([R_prime + m, G_prime + m, B_prime + m], axis=-1)
    return np.clip(rgb_img, 0.0, 1.0)


# ====================================================
# 2. CONVERSIÓN RGB <-> LCh
# ====================================================
def rgb_to_lch(img: np.ndarray) -> np.ndarray:
    """Convierte RGB [0, 1] a LCh (L [0, 100], C [0, ~100+], h [0, 360])."""
    img_norm = img.astype(np.float64) / np.max(img)
    lab = color.rgb2lab(img_norm)

    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    C = np.sqrt(a**2 + b**2)
    h = np.degrees(np.arctan2(b, a)) % 360.0

    return np.stack([L, C, h], axis=-1)


def lch_to_rgb(lch_img: np.ndarray) -> np.ndarray:
    """Reconvierte de LCh a RGB [0, 1]."""
    L, C, h = lch_img[:, :, 0], lch_img[:, :, 1], lch_img[:, :, 2]

    h_rad = np.radians(h)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)

    lab = np.stack([L, a, b], axis=-1)
    rgb_img = color.lab2rgb(lab)
    return np.clip(rgb_img, 0.0, 1.0)


# ====================================================
# 3. FUNCIONES AUXILIARES (g_m e Interpolación)
# ====================================================
def funcion_gm(componente, factor_m, es_lch=False):
    """
    g_m(x) = m * x
    - m = 1.0 : Neutro
    - m > 1.0 : Amplifica
    - 0 <= m < 1.0 : Atenúa
    """
    resultado = componente * factor_m
    return np.maximum(0.0, resultado) if es_lch else np.clip(resultado, 0.0, 1.0)


def interpolar_factores_m(tonos_imagen, puntos_control):
    """Interpolación lineal periódica para el tono [0, 360]."""
    tonos = np.array([p[0] for p in puntos_control])
    factores = np.array([p[1] for p in puntos_control])

    tonos_ext = np.concatenate([tonos - 360.0, tonos, tonos + 360.0])
    factores_ext = np.concatenate([factores, factores, factores])

    orden = np.argsort(tonos_ext)
    return np.interp(tonos_imagen, tonos_ext[orden], factores_ext[orden])


# ====================================================
# 4. HERRAMIENTA PRINCIPAL (ColorSaturation)
# ====================================================
def color_saturation(img_rgb, puntos_control, modo='HS'):
    """Modifica la saturación o croma selectivamente según el tono."""
    img_norm = img_rgb.astype(np.float64) / np.max(img_rgb)

    if modo == 'HS':
        hsv = rgb_to_hsv(img_norm)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        m_pixel = interpolar_factores_m(H, puntos_control)
        S_nueva = funcion_gm(S, m_pixel, es_lch=False)

        hsv_mod = np.stack([H, S_nueva, V], axis=-1)
        return hsv_to_rgb(hsv_mod)

    elif modo == 'LCH':
        lch = rgb_to_lch(img_norm)
        L, C, h = lch[:, :, 0], lch[:, :, 1], lch[:, :, 2]

        m_pixel = interpolar_factores_m(h, puntos_control)
        C_nuevo = funcion_gm(C, m_pixel, es_lch=True)

        lch_mod = np.stack([L, C_nuevo, h], axis=-1)
        return lch_to_rgb(lch_mod)
    else:
        raise ValueError("El modo debe ser 'HS' o 'LCh'.")


# ====================================================
# 5. BLOQUE DE EJECUCIÓN
# ====================================================
if __name__ == '__main__':
    archivo_imagen = 'P3_IMG_2387_crop.tif'
    imag = io.imread(archivo_imagen)

    # Normalización usando np.max(imag)
    img_norm = imag.astype(np.float64) / np.max(imag)

    # Puntos de control: (Tono en grados [0-360], Factor m)
    puntos_control = [(0.0, 1.0), (60.0, 2.5), (200.0, 0.1), (360.0, 1.0)]

    # Procesamiento en ambos modos
    res_hs = color_saturation(img_norm, puntos_control, modo='HS')
    res_lch = color_saturation(img_norm, puntos_control, modo='LCH')

    # Visualización comparativa
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_norm)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(res_hs)
    axes[1].set_title("Resultado Modo HS")
    axes[1].axis('off')

    axes[2].imshow(res_lch)
    axes[2].set_title("Resultado Modo LCh")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('resultado_pregunta1.png', dpi=300)
    plt.show()
    print("¡Proceso completado con éxito!")