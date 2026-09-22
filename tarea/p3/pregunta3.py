import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def cargar_imagen_tif(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No pillé el archivo: {ruta}")
        
    img = Image.open(ruta)
    img_np = np.array(img)
    
    # Si viene en RGBA, botamos el canal alfa para trabajar solo con RGB
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
        
    return img_np


def reescalar_imagen(img, s, modo='bilinear'):
    
    if s < 0.5 or s > 2.0:
        raise ValueError("El factor de escala s debe estar entre 0.5 y 2.0")
        
    # Guardamos si es escala de grises para devolverla en el mismo formato al final
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
        i_nearest = np.round(y_in).astype(int)
        j_nearest = np.round(x_in).astype(int)
        
        i_nearest = np.clip(i_nearest, 0, h_in - 1)
        j_nearest = np.clip(j_nearest, 0, w_in - 1)
        
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
        raise ValueError("Modo no reconocido, usa 'nearest' o 'bilinear'")

    if es_grises:
        salida = salida[:, :, 0]
        
    return salida


def crear_tif_sintetica(nombre="imagen_test.tif"):
    h, w = 240, 240
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            if (i // 10 + j // 10) % 2 == 0:
                img[i, j] = [180, 180, 180]
                
    # Añadimos líneas finas y cruzadas
    np.fill_diagonal(img[:, :, 0], 255)
    img[120, :, 1] = 255
    img[:, 120, 2] = 255
    
    img_pil = Image.fromarray(img)
    img_pil.save(nombre)
    print(f"-> Imagen sintética creada: {nombre}")


def correr_experimentos(archivo_tif):
    print(f"Cargando {archivo_tif}...")
    img = cargar_imagen_tif(archivo_tif)
    print(f"Imagen cargada. Tamaño: {img.shape}, Tipo: {img.dtype}")

    factores = [0.63, 0.85, 1.37, 1.74]
    
    fig, axes = plt.subplots(len(factores), 3, figsize=(11, 12))
    fig.suptitle("Comparación Nearest vs Bilinear", fontsize=12)

    for idx, s in enumerate(factores):
        res_nn = reescalar_imagen(img, s=s, modo='nearest')
        res_bl = reescalar_imagen(img, s=s, modo='bilinear')

        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title(f"Original ({img.shape[0]}x{img.shape[1]})")
        
        axes[idx, 1].imshow(res_nn)
        axes[idx, 1].set_title(f"Nearest (s={s})")
        
        axes[idx, 2].imshow(res_bl)
        axes[idx, 2].set_title(f"Bilinear (s={s})")

        for ax in axes[idx]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig("exp_1_al_4_comparacion.png")
    plt.close()
    print("-> Guardado: exp_1_al_4_comparacion.png")

    s_down = 0.55
    s_up = 1.0 / s_down 
    
    img_chica = reescalar_imagen(img, s=s_down, modo='bilinear')
    img_recuperada = reescalar_imagen(img_chica, s=s_up, modo='bilinear')
    
    h_min = min(img.shape[0], img_recuperada.shape[0])
    w_min = min(img.shape[1], img_recuperada.shape[1])
    diferencia = np.abs(img[:h_min, :w_min].astype(float) - img_recuperada[:h_min, :w_min].astype(float))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(img); axes[0].set_title("Original")
    axes[1].imshow(img_chica); axes[1].set_title(f"Achicada (s={s_down})")
    axes[2].imshow(img_recuperada); axes[2].set_title(f"Reconstruida (s={s_up:.2f})")
    
    if diferencia.ndim == 3:
        axes[3].imshow(diferencia.mean(axis=2), cmap='hot')
    else:
        axes[3].imshow(diferencia, cmap='hot')
    axes[3].set_title("Pérdida de detalle")
    
    for ax in axes: 
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("exp_5_reduccion_reconstruccion.png")
    plt.close()
    print("-> Guardado: exp_5_reduccion_reconstruccion.png")

    s1, s2 = 0.72, 0.81
    s_directo = s1 * s2
    
    paso1 = reescalar_imagen(img, s=s1, modo='bilinear')
    res_sucesivo = reescalar_imagen(paso1, s=s2, modo='bilinear')
    
    res_directo = reescalar_imagen(img, s=s_directo, modo='bilinear')
    
    h_m = min(res_sucesivo.shape[0], res_directo.shape[0])
    w_m = min(res_sucesivo.shape[1], res_directo.shape[1])
    
    diff_pasos = np.abs(res_sucesivo[:h_m, :w_m].astype(float) - res_directo[:h_m, :w_m].astype(float))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(res_sucesivo); axes[0].set_title(f"Sucesivo (s1={s1}, s2={s2})")
    axes[1].imshow(res_directo); axes[1].set_title(f"Directo (s={s_directo:.3f})")
    
    if diff_pasos.ndim == 3:
        axes[2].imshow(diff_pasos.mean(axis=2), cmap='magma')
    else:
        axes[2].imshow(diff_pasos, cmap='magma')
    axes[2].set_title("Diferencia entre métodos")
    
    for ax in axes: 
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("exp_6_sucesivo_vs_directo.png")
    plt.close()
    print("-> Guardado: exp_6_sucesivo_vs_directo.png")



if __name__ == "__main__":
    archivo_entrada = "P3_IMG_2387_crop.tif"
    
    if not os.path.exists(archivo_entrada):
        crear_tif_sintetica(archivo_entrada)
        
    correr_experimentos(archivo_entrada)