import numpy as np
from skimage import io, color, img_as_ubyte, exposure
import matplotlib.pyplot as plt

# ==========================================
# 1. FUNCIONES DE PROCESAMIENTO
# ==========================================

def ecualizacion_global(img_u8):
    """Ecualización global clásica."""
    hist, _ = np.histogram(img_u8.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_norm = np.round(255 * cdf / cdf[-1]).astype(np.uint8)
    return cdf_norm[img_u8]

def recortar_y_redistribuir_histograma(hist, clip_limit):
    """Mecanismo propio de control de contraste."""
    if clip_limit <= 0:
        return hist
    total_bins = len(hist)
    exceso = 0.0
    for i in range(total_bins):
        if hist[i] > clip_limit:
            exceso += (hist[i] - clip_limit)
            hist[i] = clip_limit
    
    exceso_int = int(exceso)
    incremento = exceso_int // total_bins
    resto = int(exceso_int % total_bins)
    
    hist += incremento
    for i in range(resto):
        hist[i] += 1
    return hist

def ecualizacion_local_malla(img_u8, tile_size=(64, 64), stride=32, clip_limit=0.0):
    """
    Ecualización local por malla con interpolación bilineal.
    Si clip_limit = 0, actúa como local no limitada.
    Si clip_limit > 0, aplica el control de contraste propio.
    """
    H, W = img_u8.shape
    th, tw = tile_size

    y_centers = list(range(th // 2, H, stride))
    x_centers = list(range(tw // 2, W, stride))
    if y_centers[-1] != H - 1: y_centers.append(H - 1)
    if x_centers[-1] != W - 1: x_centers.append(W - 1)

    cdf_list = []
    for yc in y_centers:
        row_cdfs = []
        for xc in x_centers:
            y1 = max(0, yc - th // 2)
            y2 = min(H, yc + th // 2)
            x1 = max(0, xc - tw // 2)
            x2 = min(W, xc + tw // 2)
            
            patch = img_u8[y1:y2, x1:x2]
            hist, _ = np.histogram(patch.flatten(), bins=256, range=(0, 256))
            
            if clip_limit > 0:
                hist = recortar_y_redistribuir_histograma(hist.astype(np.float64), clip_limit)
                
            cdf = hist.cumsum()
            if cdf[-1] > 0:
                cdf_norm = 255.0 * cdf / cdf[-1]
            else:
                cdf_norm = np.arange(256, dtype=np.float64)
            row_cdfs.append(cdf_norm)
        cdf_list.append(row_cdfs)

    salida = np.zeros_like(img_u8, dtype=np.float64)
    for y in range(H):
        y_idx = 0
        while y_idx < len(y_centers) - 1 and y_centers[y_idx + 1] <= y:
            y_idx += 1
        y0, y1 = y_centers[max(0, y_idx)], y_centers[min(len(y_centers) - 1, y_idx + 1)]

        for x in range(W):
            x_idx = 0
            while x_idx < len(x_centers) - 1 and x_centers[x_idx + 1] <= x:
                x_idx += 1
            x0, x1 = x_centers[max(0, x_idx)], x_centers[min(len(x_centers) - 1, x_idx + 1)]

            val = img_u8[y, x]
            if y0 == y1 and x0 == x1:
                salida[y, x] = cdf_list[y_idx][x_idx][val]
            elif y0 == y1:
                factor = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
                salida[y, x] = (1.0 - factor) * cdf_list[y_idx][x_idx][val] + factor * cdf_list[y_idx][x_idx + 1][val]
            elif x0 == x1:
                factor = (y - y0) / (y1 - y0) if y1 != y0 else 0.0
                salida[y, x] = (1.0 - factor) * cdf_list[y_idx][x_idx][val] + factor * cdf_list[y_idx + 1][x_idx][val]
            else:
                r1 = ((x1 - x) / (x1 - x0)) * cdf_list[y_idx][x_idx][val] + ((x - x0) / (x1 - x0)) * cdf_list[y_idx][x_idx + 1][val]
                r2 = ((x1 - x) / (x1 - x0)) * cdf_list[y_idx + 1][x_idx][val] + ((x - x0) / (x1 - x0)) * cdf_list[y_idx + 1][x_idx + 1][val]
                factor_y = (y - y0) / (y1 - y0)
                salida[y, x] = (1.0 - factor_y) * r1 + factor_y * r2

    return np.clip(salida, 0, 255).astype(np.uint8)


# ==========================================
# 2. EJECUCIÓN Y GENERACIÓN DE LA FIGURA
# ==========================================
if __name__ == '__main__':
    # Cargar imagen base
    archivo = 'P2_IMG_2423.tif'
    img_rgb = io.imread(archivo)
    
    # Convertir a escala de grises [0, 255]
    if img_rgb.ndim == 3:
        img_gray = color.rgb2gray(img_rgb)
    else:
        img_gray = img_rgb.astype(np.float64) / np.max(img_rgb)
    img_u8 = img_as_ubyte(img_gray)

    # Parámetros de la malla comunes
    t_size = (64, 64)
    s_val = 32
    c_limit = 30.0

    print("Calculando Métodos...")
    # 1. Global
    img_global = ecualizacion_global(img_u8)
    
    # 2. Local No Limitada (clip_limit = 0.0)
    img_local_libres = ecualizacion_local_malla(img_u8, tile_size=t_size, stride=s_val, clip_limit=0.0)
    
    # 3. Propuesta con Control de Contraste
    img_propuesta = ecualizacion_local_malla(img_u8, tile_size=t_size, stride=s_val, clip_limit=c_limit)
    
    # 4. CLAHE de referencia (Scikit-Image)
    img_clahe = exposure.equalize_adapthist(img_gray, kernel_size=64, clip_limit=0.03)
    img_clahe_u8 = img_as_ubyte(img_clahe)

    # Generar visualización conjunta de los 5 paneles
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    axes[0].imshow(img_u8, cmap='gray')
    axes[0].set_title("1. Original")
    axes[0].axis('off')

    axes[1].imshow(img_global, cmap='gray')
    axes[1].set_title("2. Global Clásica")
    axes[1].axis('off')

    axes[2].imshow(img_local_libres, cmap='gray')
    axes[2].set_title(f"3. Local No Limitada\n(Tile: {t_size}, Stride: {s_val})")
    axes[2].axis('off')

    axes[3].imshow(img_propuesta, cmap='gray')
    axes[3].set_title(f"4. Propuesta Propia\n(Clip Limit: {c_limit})")
    axes[3].axis('off')

    axes[4].imshow(img_clahe_u8, cmap='gray')
    axes[4].set_title("5. CLAHE (Referencia)\n(Scikit-Image)")
    axes[4].axis('off')

    plt.tight_layout()
    nombre_salida = 'comparacion_final_pregunta2.png'
    plt.savefig(nombre_salida, bbox_inches='tight', dpi=300)
    print(f"¡Imagen comparativa de 5 paneles generada y guardada como '{nombre_salida}'!")