import numpy as np
from skimage import io, color, img_as_ubyte

def ecualizacion_local_malla(img_rgb, tile_size=(64, 64), stride=32):
    """
    Ecualización local de histograma basada en una malla de regiones con interpolación bilineal.
    """
    # 1. Convertir a escala de grises [0, 255]
    if img_rgb.ndim == 3:
        img_gray = color.rgb2gray(img_rgb)
    else:
        img_gray = img_rgb.astype(np.float64) / np.max(img_rgb)
    
    img_u8 = img_as_ubyte(img_gray)
    H, W = img_u8.shape
    th, tw = tile_size

    # Si el tamaño del tile cubre toda la imagen, actúa como ecualización global
    if th >= H and tw >= W:
        hist, _ = np.histogram(img_u8.flatten(), bins=256, range=(0, 256))
        cdf = hist.cumsum()
        cdf_normalized = np.round(255 * cdf / cdf[-1]).astype(np.uint8)
        return cdf_normalized[img_u8]

    # 2. Generar centros de la malla
    y_centers = list(range(th // 2, H, stride))
    x_centers = list(range(tw // 2, W, stride))
    
    # Asegurar que los bordes extremos estén cubiertos
    if y_centers[-1] != H - 1:
        y_centers.append(H - 1)
    if x_centers[-1] != W - 1:
        x_centers.append(W - 1)

    # 3. Calcular la CDF para cada región (tile) en la malla
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
            cdf = hist.cumsum()
            if cdf[-1] > 0:
                cdf_norm = 255.0 * cdf / cdf[-1]
            else:
                cdf_norm = np.arange(256)
            row_cdfs.append(cdf_norm)
        cdf_list.append(row_cdfs)

    # 4. Reconstrucción mediante interpolación bilineal por píxel
    salida = np.zeros_like(img_u8, dtype=np.float64)
    
    for y in range(H):
        # Encontrar los índices de los centros en Y
        y_idx = 0
        while y_idx < len(y_centers) - 1 and y_centers[y_idx + 1] <= y:
            y_idx += 1
        y0, y1 = y_centers[max(0, y_idx)], y_centers[min(len(y_centers) - 1, y_idx + 1)]

        for x in range(W):
            # Encontrar los índices de los centros en X
            x_idx = 0
            while x_idx < len(x_centers) - 1 and x_centers[x_idx + 1] <= x:
                x_idx += 1
            x0, x1 = x_centers[max(0, x_idx)], x_centers[min(len(x_centers) - 1, x_idx + 1)]

            val = img_u8[y, x]

            if y0 == y1 and x0 == x1:
                salida[y, x] = cdf_list[y_idx][x_idx][val]
            elif y0 == y1:
                # Interpolación lineal horizontal
                factor = (x - x0) / (x1 - x0) if x1 != x0 else 0
                v0 = cdf_list[y_idx][x_idx][val]
                v1 = cdf_list[y_idx][x_idx + 1][val]
                salida[y, x] = (1 - factor) * v0 + factor * v1
            elif x0 == x1:
                # Interpolación lineal vertical
                factor = (y - y0) / (y1 - y0) if y1 != y0 else 0
                v0 = cdf_list[y_idx][x_idx][val]
                v1 = cdf_list[y_idx + 1][x_idx][val]
                salida[y, x] = (1 - factor) * v0 + factor * v1
            else:
                # Interpolación bilineal completa de las 4 celdas vecinas
                r1 = (x1 - x) / (x1 - x0) * cdf_list[y_idx][x_idx][val] + (x - x0) / (x1 - x0) * cdf_list[y_idx][x_idx + 1][val]
                r2 = (x1 - x) / (x1 - x0) * cdf_list[y_idx + 1][x_idx][val] + (x - x0) / (x1 - x0) * cdf_list[y_idx + 1][x_idx + 1][val]
                factor_y = (y - y0) / (y1 - y0)
                salida[y, x] = (1 - factor_y) * r1 + factor_y * r2

    return np.clip(salida, 0, 255).astype(np.uint8)

# Bloque de ejecución principal
if __name__ == '__main__':
    imagen = io.imread('P2_IMG_2423.tif')
    resultado_local = ecualizacion_local_malla(imagen, tile_size=(64, 64), stride=32)
    io.imsave('resultado_pregunta2_local.png', resultado_local)
    print("¡Ecualización local completada y guardada exitosamente!")