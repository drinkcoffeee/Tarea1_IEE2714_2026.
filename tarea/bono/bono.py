import numpy as np

# ---------------------------------------------------------------------
# 1. GENERACIÓN DE MOSAICO BAYER (RGGB) Y SUPER-PIXEL
# ---------------------------------------------------------------------

def armar_mosaico_bayer(img_rgb):
    """
    Simula el sensor RGGB filtrando un solo canal por cada píxel.
    Patrón 2x2:
      [ R , G1 ]
      [ G2,  B ]
    """
    alto, ancho, _ = img_rgb.shape
    bayer = np.zeros((alto, ancho), dtype=img_rgb.dtype)
    
    # R en filas pares, col pares
    bayer[0::2, 0::2] = img_rgb[0::2, 0::2, 0]
    # G1 en filas pares, col impares
    bayer[0::2, 1::2] = img_rgb[0::2, 1::2, 1]
    # G2 en filas impares, col pares
    bayer[1::2, 0::2] = img_rgb[1::2, 0::2, 1]
    # B en filas impares, col impares
    bayer[1::2, 1::2] = img_rgb[1::2, 1::2, 2]
    
    return bayer


def generar_super_pixel(bayer):
    """
    Procesa bloques 2x2 RGGB para armar una imagen a mitad de tamaño.
    G se obtiene promediando los dos sensores verdes (G1 y G2).
    """
    alto, ancho = bayer.shape
    h_sub, w_sub = alto // 2, ancho // 2
    salida = np.zeros((h_sub, w_sub, 3), dtype=np.float32)
    
    # Extraemos cada posición del patrón 2x2
    r  = bayer[0:2*h_sub:2, 0:2*w_sub:2]
    g1 = bayer[0:2*h_sub:2, 1:2*w_sub:2]
    g2 = bayer[1:2*h_sub:2, 0:2*w_sub:2]
    b  = bayer[1:2*h_sub:2, 1:2*w_sub:2]
    
    salida[:, :, 0] = r
    salida[:, :, 1] = (g1.astype(np.float32) + g2.astype(np.float32)) / 2.0
    salida[:, :, 2] = b
    
    return salida


# ---------------------------------------------------------------------
# 2. DEBAYERIZADO BILINEAL (Mantiene tamaño original)
# ---------------------------------------------------------------------

def debayer_bilineal(bayer):
    """
    Reconstruye los canales faltantes mediante promedio de vecinos directos.
    """
    alto, ancho = bayer.shape
    
    # Agregamos padding por reflexión para no complicarnos con los bordes
    padded = np.pad(bayer, pad_width=1, mode='reflect').astype(np.float32)
    rgb = np.zeros((alto, ancho, 3), dtype=np.float32)
    
    for i in range(alto):
        r_pad = i + 1
        for j in range(ancho):
            c_pad = j + 1
            
            # Posición R (fila par, col par)
            if i % 2 == 0 and j % 2 == 0:
                r_val = padded[r_pad, c_pad]
                g_val = (padded[r_pad-1, c_pad] + padded[r_pad+1, c_pad] +
                         padded[r_pad, c_pad-1] + padded[r_pad, c_pad+1]) / 4.0
                b_val = (padded[r_pad-1, c_pad-1] + padded[r_pad-1, c_pad+1] +
                         padded[r_pad+1, c_pad-1] + padded[r_pad+1, c_pad+1]) / 4.0
                
            # Posición G1 (fila par, col impar)
            elif i % 2 == 0 and j % 2 == 1:
                g_val = padded[r_pad, c_pad]
                r_val = (padded[r_pad, c_pad-1] + padded[r_pad, c_pad+1]) / 2.0
                b_val = (padded[r_pad-1, c_pad] + padded[r_pad+1, c_pad]) / 2.0
                
            # Posición G2 (fila impar, col par)
            elif i % 2 == 1 and j % 2 == 0:
                g_val = padded[r_pad, c_pad]
                r_val = (padded[r_pad-1, c_pad] + padded[r_pad+1, c_pad]) / 2.0
                b_val = (padded[r_pad, c_pad-1] + padded[r_pad, c_pad+1]) / 2.0
                
            # Posición B (fila impar, col impar)
            else:
                b_val = padded[r_pad, c_pad]
                g_val = (padded[r_pad-1, c_pad] + padded[r_pad+1, c_pad] +
                         padded[r_pad, c_pad-1] + padded[r_pad, c_pad+1]) / 4.0
                r_val = (padded[r_pad-1, c_pad-1] + padded[r_pad-1, c_pad+1] +
                         padded[r_pad+1, c_pad-1] + padded[r_pad+1, c_pad+1]) / 4.0
                
            rgb[i, j] = [r_val, g_val, b_val]
            
    return np.clip(rgb, 0, 255)


# ---------------------------------------------------------------------
# 3. REESCALADO (Inspirado en la Pregunta 3)
# ---------------------------------------------------------------------

def reescalar(img, s, modo='bilineal'):
    """
    Aumenta o reduce la imagen por un factor s usando vecino o bilineal.
    """
    h_in, w_in = img.shape[:2]
    h_out = int(np.round(h_in * s))
    w_out = int(np.round(w_in * s))
    
    es_color = (img.ndim == 3)
    canales = img.shape[2] if es_color else 1
    
    salida = np.zeros((h_out, w_out, canales) if es_color else (h_out, w_out), dtype=np.float32)
    
    for i in range(h_out):
        y = (i + 0.5) / s - 0.5
        for j in range(w_out):
            x = (j + 0.5) / s - 0.5
            
            if modo == 'vecino':
                r_near = int(np.clip(np.round(y), 0, h_in - 1))
                c_near = int(np.clip(np.round(x), 0, w_in - 1))
                salida[i, j] = img[r_near, c_near]
                
            elif modo == 'bilineal':
                r0 = int(np.floor(y))
                c0 = int(np.floor(x))
                r1, c1 = r0 + 1, c0 + 1
                
                dy, dx = y - r0, x - c0
                
                # Aseguramos límites
                r0_c, r1_c = np.clip(r0, 0, h_in - 1), np.clip(r1, 0, h_in - 1)
                c0_c, c1_c = np.clip(c0, 0, w_in - 1), np.clip(c1, 0, w_in - 1)
                
                v00 = img[r0_c, c0_c]
                v01 = img[r0_c, c1_c]
                v10 = img[r1_c, c0_c]
                v11 = img[r1_c, c1_c]
                
                # Ponderación bilineal
                val = ((1 - dx) * (1 - dy) * v00 +
                       dx * (1 - dy) * v01 +
                       (1 - dx) * dy * v10 +
                       dx * dy * v11)
                salida[i, j] = val
                
    return np.clip(salida, 0, 255)