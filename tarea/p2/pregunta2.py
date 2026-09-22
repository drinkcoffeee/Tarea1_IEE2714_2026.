import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import cv2


def cargar_imagen(ruta_imagen: str) -> np.ndarray:
   
    img_raw = io.imread(ruta_imagen)
    if img_raw.ndim == 3:
        if img_raw.shape[2] == 4:
            img_raw = img_raw[:, :, :3]
        img_gray = color.rgb2gray(img_raw)
    else:
        img_gray = img_raw.astype(np.float64)
    
    # Normalización robusta a uint8 [0, 255]
    min_val, max_val = np.min(img_gray), np.max(img_gray)
    if max_val > min_val:
        img_uint8 = (255.0 * (img_gray - min_val) / (max_val - min_val)).astype(np.uint8)
    else:
        img_uint8 = np.zeros_like(img_gray, dtype=np.uint8)
        
    return img_uint8


def calcular_cdf_region(region: np.ndarray, nbins: int = 256, clip_limit: float = None) -> np.ndarray:
    
    hist, _ = np.histogram(region.ravel(), bins=nbins, range=(0, 256))
    
    if clip_limit is not None and clip_limit > 0:
        limit = int(clip_limit * (region.size / nbins))
        limit = max(1, limit)
        
        exceso = np.maximum(hist - limit, 0)
        hist = np.minimum(hist, limit)
        
        redistribucion = exceso.sum() // nbins
        resto = exceso.sum() % nbins
        hist += redistribucion
        hist[:resto] += 1

    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
    denominador = region.size - cdf_min
    
    if denominador <= 0:
        return np.arange(nbins, dtype=np.uint8)
    
    T = np.round((cdf - cdf_min) / denominador * (nbins - 1))
    return np.clip(T, 0, nbins - 1).astype(np.uint8)

def ecualizacion_local_malla(img: np.ndarray, 
                                region_size: tuple = (64, 64), 
                                step_size: tuple = (32, 32), 
                                nbins: int = 256, 
                                clip_limit: float = None) -> np.ndarray:
 
    img_gray = img.copy()
    H, W = img_gray.shape
    Rh, Rw = region_size
    Dh, Dw = step_size

    if Rh >= H and Rw >= W:
        T_global = calcular_cdf_region(img_gray, nbins=nbins, clip_limit=None)
        return T_global[img_gray]

    centers_y = np.arange(Rh // 2, H, Dh)
    centers_x = np.arange(Rw // 2, W, Dw)
    
    if len(centers_y) == 0 or centers_y[-1] < H - Rh // 2:
        centers_y = np.append(centers_y, H - Rh // 2) if len(centers_y) > 0 else np.array([H // 2])
    if len(centers_x) == 0 or centers_x[-1] < W - Rw // 2:
        centers_x = np.append(centers_x, W - Rw // 2) if len(centers_x) > 0 else np.array([W // 2])

    n_rows = len(centers_y)
    n_cols = len(centers_x)


    cdfs = np.zeros((n_rows, n_cols, nbins), dtype=np.uint8)
    for r in range(n_rows):
        cy = centers_y[r]
        r0 = max(0, cy - Rh // 2)
        r1 = min(H, cy + Rh // 2)
        for c in range(n_cols):
            cx = centers_x[c]
            c0 = max(0, cx - Rw // 2)
            c1 = min(W, cx + Rw // 2)
            
            tile = img_gray[r0:r1, c0:c1]
            cdfs[r, c] = calcular_cdf_region(tile, nbins=nbins, clip_limit=clip_limit)

    img_out = np.zeros_like(img_gray)

    for y in range(H):
        r_idx = np.searchsorted(centers_y, y) - 1
        r0 = int(np.clip(r_idx, 0, n_rows - 2))
        r1 = r0 + 1

        dy = (y - centers_y[r0]) / (centers_y[r1] - centers_y[r0] + 1e-10)
        dy = np.clip(dy, 0.0, 1.0)

        for x in range(W):
            v = img_gray[y, x]

            c_idx = np.searchsorted(centers_x, x) - 1
            c0 = int(np.clip(c_idx, 0, n_cols - 2))
            c1 = c0 + 1

            dx = (x - centers_x[c0]) / (centers_x[c1] - centers_x[c0] + 1e-10)
            dx = np.clip(dx, 0.0, 1.0)

    
            t00 = cdfs[r0, c0, v]
            t01 = cdfs[r0, c1, v]
            t10 = cdfs[r1, c0, v]
            t11 = cdfs[r1, c1, v]

           
            top = (1 - dx) * t00 + dx * t01
            bot = (1 - dx) * t10 + dx * t11
            val_final = (1 - dy) * top + dy * bot

            img_out[y, x] = np.clip(np.round(val_final), 0, 255)

    return img_out

def ejecutar_experimentos_p2(ruta_imagen: str):
    print(f"Vicky, procesando la imagen: '{ruta_imagen}'...")
    img_uint8 = cargar_imagen(ruta_imagen)
    H, W = img_uint8.shape

    print("1/4 Calculando Ecualización Global Clásica...")
    global_propia = ecualizacion_local_malla(img_uint8, region_size=(H, W), step_size=(H, W))

    print("2/4 Calculando Ecualización Local No Limitada...")
    local_nolim = ecualizacion_local_malla(img_uint8, region_size=(64, 64), step_size=(32, 32), clip_limit=None)

    print("3/4 Calculando Propuesta con Control de Contraste...")
    local_lim_propia = ecualizacion_local_malla(img_uint8, region_size=(64, 64), step_size=(32, 32), clip_limit=3.0)

    print("4/4 Calculando CLAHE de OpenCV (Referencia)...")
    clahe_cv2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_ref = clahe_cv2.apply(img_uint8)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(img_uint8, cmap='gray')
    axes[0, 0].set_title("1. Imagen Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(global_propia, cmap='gray')
    axes[0, 1].set_title("2. Global Clásica (1 Región HxW)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(local_nolim, cmap='gray')
    axes[0, 2].set_title("3. Local No Limitada\n(Region=64x64, Step=32x32)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(local_lim_propia, cmap='gray')
    axes[1, 0].set_title("4. Propuesta Limitada\n(Clip Limit = 3.0)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(clahe_ref, cmap='gray')
    axes[1, 1].set_title("5. CLAHE Benchmark (OpenCV)")
    axes[1, 1].axis('off')

    # Histograma Comparativo
    axes[1, 2].hist(local_nolim.ravel(), bins=100, alpha=0.5, label='No Limitada', color='red')
    axes[1, 2].hist(local_lim_propia.ravel(), bins=100, alpha=0.5, label='Propuesta Limitada', color='green')
    axes[1, 2].set_title("Histograma: No Limitada vs Limitada")
    axes[1, 2].set_xlabel("Nivel de Gris")
    axes[1, 2].set_ylabel("Frecuencia")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig('resultado_pregunta2.png', dpi=300)
    print("-> ¡Proceso terminado! Imagen guardada como 'resultado_pregunta2.png'.")
    plt.show()


if __name__ == '__main__':
    
    nombre_imagen = 'P2_IMG_2423.tif'
    ejecutar_experimentos_p2(nombre_imagen)