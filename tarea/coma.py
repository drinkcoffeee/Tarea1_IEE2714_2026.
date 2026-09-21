import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

# ==========================================
# SCRIPT INDEPENDIENTE DE COMPARACIÓN HS vs LCh
# ==========================================
if __name__ == '__main__':
    print("Cargando imágenes de resultados...")
    
    # 1. Cargar las dos imágenes generadas previamente
    try:
        img_hs = io.imread('resultado_pregunta1_hs.png')
        img_lch = io.imread('resultado_pregunta1_lch.png')
    except FileNotFoundError:
        print("Error: No se encontraron los archivos 'resultado_pregunta1_hs.png' o 'resultado_pregunta1_lch.png'.")
        print("Asegúrate de ejecutar primero tu script de procesamiento.")
        exit()

    # 2. Manejar el problema de canales: si tiene 4 canales (RGBA), extraer solo RGB (:3)
    if img_hs.ndim == 3 and img_hs.shape[2] == 4:
        img_hs = img_hs[:, :, :3]
    if img_lch.ndim == 3 and img_lch.shape[2] == 4:
        img_lch = img_lch[:, :, :3]

    # Normalizar ambas imágenes a formato float en el rango [0, 1]
    img_hs = img_hs.astype(np.float64) / 255.0
    img_lch = img_lch.astype(np.float64) / 255.0

    # 3. Calcular la diferencia absoluta utilizando escala de grises
    print("Calculando mapa de diferencias...")
    gray_hs = color.rgb2gray(img_hs)
    gray_lch = color.rgb2gray(img_lch)
    diferencia = np.abs(gray_hs - gray_lch)

    # 4. Generar y guardar el gráfico comparativo de 3 paneles
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(img_hs)
    axes[0].set_title("Resultado Modo HS (HSV)")
    axes[0].axis('off')

    axes[1].imshow(img_lch)
    axes[1].set_title("Resultado Modo LCh (CIELAB)")
    axes[1].axis('off')

    # Mapa de calor con la diferencia absoluta
    im = axes[2].imshow(diferencia, cmap='inferno')
    axes[2].set_title("Mapa de Diferencias (HS vs LCh)")
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    nombre_salida = 'comparacion_hs_lch.png'
    plt.savefig(nombre_salida, bbox_inches='tight', dpi=300)
    print(f"¡Comparativa generada con éxito! Guardada como '{nombre_salida}'.")