from analisis1 import color_saturation, rgb_to_hsv, interpolar_factores_m
from skimage import io, color
import numpy as np

archivo_imagen = 'P1_IMG_2402.tif'
imag = io.imread(archivo_imagen)

    # Normalización usando np.max(imag)
img_norm = imag.astype(np.float64) / np.max(imag)
r, c = 100, 100
puntos = [(0, 1.0), (60, 2.5), (200, 0.1), (360, 1.0)]

hsv = rgb_to_hsv(img_norm)
print("RGB Orig:", img_norm[r, c].round(4))
print("H:", round(hsv[r, c, 0], 2), "S orig:", round(hsv[r, c, 1], 4))
print("m(H):", round(interpolar_factores_m(hsv[r, c, 0], puntos), 4))
print("RGB HS Final:", color_saturation(img_norm, puntos, modo='HS')[r, c].round(4))
print("RGB LCh Final:", color_saturation(img_norm, puntos, modo='LCH')[r, c].round(4))