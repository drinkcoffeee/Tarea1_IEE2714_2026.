import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONES MATEMÁTICAS COMPARTIDAS
# ==========================================
def funcion_gm(S, m):
    """Función g_m basada en potencias acotada en [0, 1]."""
    S_mod = np.power(S, m)
    return np.clip(S_mod, 0.0, 1.0)

def interpolacion_periodica_h(H, control_points):
    """Interpolación lineal periódica para el tono H o h [0, 360]."""
    h_vals = np.array([p[0] for p in control_points])
    m_vals = np.array([p[1] for p in control_points])
    
    # Extensión periódica para evitar saltos en el rojo (0° / 360°)
    h_ext = np.concatenate([h_vals - 360.0, h_vals, h_vals + 360.0])
    m_ext = np.concatenate([m_vals, m_vals, m_vals])
    
    sort_idx = np.argsort(h_ext)
    h_ext = h_ext[sort_idx]
    m_ext = m_ext[sort_idx]
    
    return np.interp(H, h_ext, m_ext)


# ==========================================
# 2. CONVERSIONES Y PROCESAMIENTO MODO HS
# ==========================================
def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    img = np.array(img).astype(np.float64) / np.max(img)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
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
    X = C * (1 - np.abs((H / 60.0) % 2 - 1))
    m = V - C

    R_prime = np.zeros_like(H)
    G_prime = np.zeros_like(H)
    B_prime = np.zeros_like(H)

    idx = (H >= 0) & (H < 60)
    R_prime[idx], G_prime[idx], B_prime[idx] = C[idx], X[idx], 0
    idx = (H >= 60) & (H < 120)
    R_prime[idx], G_prime[idx], B_prime[idx] = X[idx], C[idx], 0
    idx = (H >= 120) & (H < 180)
    R_prime[idx], G_prime[idx], B_prime[idx] = 0, C[idx], X[idx]
    idx = (H >= 180) & (H < 240)
    R_prime[idx], G_prime[idx], B_prime[idx] = 0, X[idx], C[idx]
    idx = (H >= 240) & (H < 300)
    R_prime[idx], G_prime[idx], B_prime[idx] = X[idx], 0, C[idx]
    idx = (H >= 300) & (H <= 360)
    R_prime[idx], G_prime[idx], B_prime[idx] = C[idx], 0, X[idx]

    rgb_img = np.stack([R_prime + m, G_prime + m, B_prime + m], axis=-1)
    return np.clip(rgb_img, 0.0, 1.0)

def procesar_hs(imag, control_points):
    hsv_img = rgb_to_hsv(imag)
    H, S, V = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    m_map = interpolacion_periodica_h(H, control_points)
    S_prime = funcion_gm(S, m_map)
    hsv_mod = np.stack([H, S_prime, V], axis=-1)
    return hsv_to_rgb(hsv_mod)


# ==========================================
# 3. CONVERSIONES Y PROCESAMIENTO MODO LCh
# ==========================================
def rgb_to_lch(img: np.ndarray) -> np.ndarray:
    lab = color.rgb2lab(img)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    C = np.sqrt(a ** 2 + b ** 2)
    h = np.degrees(np.arctan2(b, a))
    h = np.where(h < 0, h + 360.0, h)
    return np.stack([L, C, h], axis=-1)

def lch_to_rgb(lch_img: np.ndarray) -> np.ndarray:
    L, C, h = lch_img[:, :, 0], lch_img[:, :, 1], lch_img[:, :, 2]
    h_rad = np.radians(h)
    a = C * np.cos(h_rad)
    b = C * np.sin(h_rad)
    lab = np.stack([L, a, b], axis=-1)
    rgb_img = color.lab2rgb(lab)
    return np.clip(rgb_img, 0.0, 1.0)

def procesar_lch(imag, control_points):
    lch_img = rgb_to_lch(imag)
    L, C, h = lch_img[:, :, 0], lch_img[:, :, 1], lch_img[:, :, 2]
    m_map = interpolacion_periodica_h(h, control_points)
    
    # Normalización estable del croma para g_m
    C_max_val = np.max(C) if np.max(C) > 0 else 1.0
    C_prime = funcion_gm(C / C_max_val, m_map) * C_max_val
    
    lch_mod = np.stack([L, C_prime, h], axis=-1)
    return lch_to_rgb(lch_mod)


# ==========================================
# 4. BLOQUE PRINCIPAL DE COMPARACIÓN
# ==========================================
if __name__ == '__main__':
    # Cargar imagen de prueba
    archivo_imagen = 'P1_IMG_2402.tif'
    imag = io.imread(archivo_imagen)
    rgb_original = np.array(imag).astype(np.float64) / np.max(imag)

    # DEFINIR EL MAPEO M(H) EQUIVALENTE PARA AMBOS ESPACIOS
    # (Mismos puntos de control exactos)
    puntos_control_equivalentes = [
        (0.0, 1.0), 
        (60.0, 2.5),   # Amplificación fuerte en amarillos
        (200.0, 0.2),  # Atenuación fuerte en cianes/azules
        (360.0, 1.0)
    ]

    print("Calculando procesamiento con mapeo equivalente en modo HS...")
    rgb_hs = procesar_hs(rgb_original, puntos_control_equivalentes)

    print("Calculando procesamiento con mapeo equivalente en modo LCh...")
    rgb_lch = procesar_lch(rgb_original, puntos_control_equivalentes)

    # Generar figura de comparación lado a lado
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rgb_original)
    axes[0].set_title("1. Imagen Original")
    axes[0].axis('off')

    axes[1].imshow(rgb_hs)
    axes[1].set_title("2. Modificación HS (Saturación)")
    axes[1].axis('off')

    axes[2].imshow(rgb_lch)
    axes[2].set_title("3. Modificación LCh (Croma)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('comparacion_directa_hs_lch.png', bbox_inches='tight', dpi=300)
    print("¡Listo! Se ha guardado la imagen 'comparacion_directa_hs_lch.png' con los resultados equivalentes.")