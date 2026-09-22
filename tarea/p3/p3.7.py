import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def cargar_imagen_tif(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
        
    img = Image.open(ruta)
    img_np = np.array(img)
    
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
        
    return img_np


def reescalar_imagen(img, s, modo='bilinear'):
    if s < 0.5 or s > 2.0:
        raise ValueError("El factor de escala s debe estar entre 0.5 y 2.0")
        
    es_grises = (img.ndim == 2)
    if es_grises:
        img_trabajo = img[:, :, np.newaxis]
    else:
        img_trabajo = img.copy()
        
    h_in, w_in, canales = img_trabajo.shape
    
    h_out = int(round(h_in * s))
    w_out = int(round(w_in * s))
    
    i_out, j_out = np.meshgrid(np.arange(h_out), np.arange(w_out), indexing='ij')
    
    y_in = (i_out + 0.5) / s - 0.5
    x_in = (j_out + 0.5) / s - 0.5
    
    if modo == 'nearest':
        i_nearest = np.clip(np.round(y_in).astype(int), 0, h_in - 1)
        j_nearest = np.clip(np.round(x_in).astype(int), 0, w_in - 1)
        salida = img_trabajo[i_nearest, j_nearest]
        
    elif modo == 'bilinear':
        i0 = np.floor(y_in).astype(int)
        i1 = i0 + 1
        j0 = np.floor(x_in).astype(int)
        j1 = j0 + 1
        
        dy = (y_in - i0)[:, :, np.newaxis]
        dx = (x_in - j0)[:, :, np.newaxis]
        
        i0_c = np.clip(i0, 0, h_in - 1)
        i1_c = np.clip(i1, 0, h_in - 1)
        j0_c = np.clip(j0, 0, w_in - 1)
        j1_c = np.clip(j1, 0, w_in - 1)
        
        v00 = img_trabajo[i0_c, j0_c].astype(float)
        v01 = img_trabajo[i0_c, j1_c].astype(float)
        v10 = img_trabajo[i1_c, j0_c].astype(float)
        v11 = img_trabajo[i1_c, j1_c].astype(float)
        
        w00 = (1.0 - dy) * (1.0 - dx)
        w01 = (1.0 - dy) * dx
        w10 = dy * (1.0 - dx)
        w11 = dy * dx
        
        val = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
        
        if np.issubdtype(img.dtype, np.integer):
            val_max = np.iinfo(img.dtype).max
            salida = np.clip(val, 0, val_max).astype(img.dtype)
        else:
            salida = val.astype(img.dtype)
    else:
        raise ValueError("Modo no válido. Usa 'nearest' o 'bilinear'.")

    if es_grises:
        salida = salida[:, :, 0]
        
    return salida



def ejecutar_experimentos(ruta_imagen, roi=None):
    print("\n" + "="*80)
    print(" EJECUTANDO EXPERIMENTOS Y GENERANDO GRÁFICOS ")
    print("="*80)
    
    img = cargar_imagen_tif(ruta_imagen)
    h_orig, w_orig = img.shape[:2]
    print(f"Imagen cargada correctamente: {h_orig}x{w_orig} píxeles | Tipo: {img.dtype}\n")

    if roi is None:
        y1, y2 = int(h_orig * 0.35), int(h_orig * 0.65)
        x1, x2 = int(w_orig * 0.35), int(w_orig * 0.65)
    else:
        y1, y2, x1, x2 = roi

    factores = [0.63, 0.85, 1.37, 1.74]
    fig, axes = plt.subplots(len(factores), 3, figsize=(11, 11))
    fig.suptitle("Experimentos 1-4: Comparación con Recortes Ampliados (ZOOM)", fontsize=12)

    for idx, s in enumerate(factores):
        res_nn = reescalar_imagen(img, s=s, modo='nearest')
        res_bl = reescalar_imagen(img, s=s, modo='bilinear')

        y1_s, y2_s = int(y1 * s), int(y2 * s)
        x1_s, x2_s = int(x1 * s), int(x2 * s)

        axes[idx, 0].imshow(img[y1:y2, x1:x2])
        axes[idx, 0].set_title(f"Original Zoom")
        
        axes[idx, 1].imshow(res_nn[y1_s:y2_s, x1_s:x2_s])
        axes[idx, 1].set_title(f"Nearest (s={s})")
        
        axes[idx, 2].imshow(res_bl[y1_s:y2_s, x1_s:x2_s])
        axes[idx, 2].set_title(f"Bilinear (s={s})")

        for ax in axes[idx]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()  

    
    s_down = 0.55
    s_up = 1.0 / s_down
    
    img_chica = reescalar_imagen(img, s=s_down, modo='bilinear')
    img_recuperada = reescalar_imagen(img_chica, s=s_up, modo='bilinear')
    
    h_m = min(img.shape[0], img_recuperada.shape[0])
    w_m = min(img.shape[1], img_recuperada.shape[1])
    diferencia = np.abs(img[:h_m, :w_m].astype(float) - img_recuperada[:h_m, :w_m].astype(float))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("Experimento 5: Reducción seguida de Ampliación", fontsize=12)
    
    axes[0].imshow(img[y1:y2, x1:x2]); axes[0].set_title("Original Zoom")
    
    y1_d, y2_d = int(y1 * s_down), int(y2 * s_down)
    x1_d, x2_d = int(x1 * s_down), int(x2 * s_down)
    
    axes[1].imshow(img_chica[y1_d:y2_d, x1_d:x2_d]); axes[1].set_title(f"Reducida (s={s_down})")
    axes[2].imshow(img_recuperada[y1:y2, x1:x2]); axes[2].set_title(f"Recuperada (s={s_up:.2f})")
    
    diff_crop = diferencia[y1:min(y2, h_m), x1:min(x2, w_m)]
    if diff_crop.ndim == 3:
        axes[3].imshow(diff_crop.mean(axis=2), cmap='hot')
    else:
        axes[3].imshow(diff_crop, cmap='hot')
    axes[3].set_title("Pérdida de Detalle")
    
    for ax in axes: 
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    s1, s2 = 0.72, 0.81
    s_directo = s1 * s2
    
    paso1 = reescalar_imagen(img, s=s1, modo='bilinear')
    res_sucesivo = reescalar_imagen(paso1, s=s2, modo='bilinear')
    res_directo = reescalar_imagen(img, s=s_directo, modo='bilinear')
    
    h_m2 = min(res_sucesivo.shape[0], res_directo.shape[0])
    w_m2 = min(res_sucesivo.shape[1], res_directo.shape[1])
    diff_pasos = np.abs(res_sucesivo[:h_m2, :w_m2].astype(float) - res_directo[:h_m2, :w_m2].astype(float))

    y1_sd, y2_sd = int(y1 * s_directo), int(y2 * s_directo)
    x1_sd, x2_sd = int(x1 * s_directo), int(x2 * s_directo)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Experimento 6: Reescalados Sucesivos vs Único Directo", fontsize=12)
    
    axes[0].imshow(res_sucesivo[y1_sd:y2_sd, x1_sd:x2_sd]); axes[0].set_title(f"Sucesivo ({s1}x{s2})")
    axes[1].imshow(res_directo[y1_sd:y2_sd, x1_sd:x2_sd]); axes[1].set_title(f"Directo (s={s_directo:.3f})")
    
    diff_pasos_crop = diff_pasos[y1_sd:min(y2_sd, h_m2), x1_sd:min(x2_sd, w_m2)]
    if diff_pasos_crop.ndim == 3:
        axes[2].imshow(diff_pasos_crop.mean(axis=2), cmap='magma')
    else:
        axes[2].imshow(diff_pasos_crop, cmap='magma')
    axes[2].set_title("Diferencia de Pixel")
    
    for ax in axes: 
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    RUTA_IMAGEN = "P3_IMG_2387_crop.tif" 
    
    ROI_ZOOM = (600, 1100, 400, 900) 
    
    ejecutar_experimentos(RUTA_IMAGEN, roi=ROI_ZOOM)
    
    