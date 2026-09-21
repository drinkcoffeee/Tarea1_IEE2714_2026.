import numpy as np
from skimage import io, color, img_as_ubyte

def recortar_y_redistribuir_histograma(hist, clip_limit):
    """
    Mecanismo propio de control de contraste inspirado en CLAHE.
    Recorta el histograma según el clip_limit y redistribuye el exceso uniformemente.
    """
    if clip_limit <= 0:
        return hist
    
    total_bins = len(hist)
    exceso = 0.0
    for i in range(total_bins):
        if hist[i] > clip_limit:
            exceso += (hist[i] - clip_limit)
            hist[i] = clip_limit
            
    # Convertir explícitamente el exceso a entero para que range() no falle
    exceso_int = int(exceso)
    incremento = exceso_int // total_bins
    resto = int(exceso_int % total_bins)
    
    hist += incremento
    for i in range(resto):
        hist[i] += 1
        
    return hist

def ecualizacion_local_con_control_contraste(img_rgb, tile_size=(64, 64), stride=32, clip_limit=40.0):
    """
    Ecualización local con malla parametrizable, interpolación bilineal 
    y control de contraste mediante recorte de histograma.
    """
    # 1. Conversión de la imagen a escala de grises [0, 255]
    if img_rgb.ndim == 3:
        img_gray = color.rgb2gray(img_rgb)
    else:
        img_gray = img_rgb.astype(np.float64) / np.max(img_rgb)
    
    img_u8 = img_as_ubyte(img_gray)
    H, W = img_u8.shape
    th, tw = tile_size

    # Modo Global si el tile cubre toda la imagen
    if th >= H and tw >= W:
        hist, _ = np.histogram(img_u8.flatten(), bins=256, range=(0, 256))
        hist = recortar_y_redistribuir_histograma(hist.astype(np.float64), clip_limit)
        cdf = hist.cumsum()
        cdf_norm = np.round(255 * cdf / cdf[-1]).astype(np.uint8)
        return cdf_norm[img_u8]

    # 2. Generación manual de los centros de la malla
    y_centers = list(range(th // 2, H, stride))
    x_centers = list(range(tw // 2, W, stride))
    
    if y_centers[-1] != H - 1:
        y_centers.append(H - 1)
    if x_centers[-1] != W - 1:
        x_centers.append(W - 1)

    # 3. Construcción de las transformaciones locales (CDFs) por región con limitación de contraste
    cdf_list = []
    for yc in y_centers:
        row_cdfs = []
        for xc in x_centers:
            y1 = max(0, yc - th // 2)
            y2 = min(H, yc + th // 2)
            x1 = max(0, xc - tw // 2)
            x2 = min(W, xc + tw // 2)
            
            patch = img_u8[y1:y2, x1:x2]
            
            # Cálculo eficiente del histograma base (permitido por enunciado)
            hist, _ = np.histogram(patch.flatten(), bins=256, range=(0, 256))
            
            # Aplicación del mecanismo de control de contraste
            hist_limitado = recortar_y_redistribuir_histograma(hist.astype(np.float64), clip_limit)
            
            # Construcción de la función de distribución acumulada (CDF)
            cdf = hist_limitado.cumsum()
            if cdf[-1] > 0:
                cdf_norm = 255.0 * cdf / cdf[-1]
            else:
                cdf_norm = np.arange(256, dtype=np.float64)
                
            row_cdfs.append(cdf_norm)
        cdf_list.append(row_cdfs)

    # 4. Combinación de transformaciones locales mediante interpolación bilineal píxel a píxel
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
                v0 = cdf_list[y_idx][x_idx][val]
                v1 = cdf_list[y_idx][x_idx + 1][val]
                salida[y, x] = (1.0 - factor) * v0 + factor * v1
            elif x0 == x1:
                factor = (y - y0) / (y1 - y0) if y1 != y0 else 0.0
                v0 = cdf_list[y_idx][x_idx][val]
                v1 = cdf_list[y_idx + 1][x_idx][val]
                salida[y, x] = (1.0 - factor) * v0 + factor * v1
            else:
                # Interpolación bilineal completa de las 4 celdas vecinas de la malla
                r1 = ((x1 - x) / (x1 - x0)) * cdf_list[y_idx][x_idx][val] + ((x - x0) / (x1 - x0)) * cdf_list[y_idx][x_idx + 1][val]
                r2 = ((x1 - x) / (x1 - x0)) * cdf_list[y_idx + 1][x_idx][val] + ((x - x0) / (x1 - x0)) * cdf_list[y_idx + 1][x_idx + 1][val]
                factor_y = (y - y0) / (y1 - y0)
                salida[y, x] = (1.0 - factor_y) * r1 + factor_y * r2

    return np.clip(salida, 0, 255).astype(np.uint8)

if __name__ == '__main__':
    # Carga de la imagen base de la tarea
    imagen = io.imread('P2_IMG_2423.tif')
    
    # Ejecución con control de contraste parametrizado (clip_limit regula la ganancia máxima)
    resultado = ecualizacion_local_con_control_contraste(imagen, tile_size=(64, 64), stride=32, clip_limit=30.0)
    
    io.imsave('resultado_pregunta2_clipping.png', resultado)
    print("¡Proceso de ecualización local con control de contraste finalizado con éxito!")