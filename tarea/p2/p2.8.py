import cv2
import numpy as np
import matplotlib.pyplot as plt


NOMBRE_IMAGEN = "P2_IMG_2423.tif" 
REGION_SIZE = (64, 64)           # Tamaño de las ventanas locales
STEP_SIZE = (32, 32)             # Separación entre centros (50% de overlap)
CLIP_LIMIT = 3.0                

img = cv2.imread(NOMBRE_IMAGEN, cv2.IMREAD_UNCHANGED)

if img is None:
    print(f"❌ Error: No encontré '{NOMBRE_IMAGEN}'. Revisa el nombre y la carpeta.")
    exit()

if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Si es de 16 bits o float, la ajustamos a 8 bits bien escalada
if img.dtype != np.uint8:
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

H, W = img.shape
rh, rw = REGION_SIZE
sh, sw = STEP_SIZE


centros_y = np.arange(rh // 2, H, sh)
centros_x = np.arange(rw // 2, W, sw)

if centros_y[-1] < H - 1: centros_y = np.append(centros_y, H - 1)
if centros_x[-1] < W - 1: centros_x = np.append(centros_x, W - 1)

ny, nx = len(centros_y), len(centros_x)


luts = np.zeros((ny, nx, 256), dtype=np.uint8)

for r in range(ny):
    for c in range(nx):
        cy, cx = centros_y[r], centros_x[c]
        
        y0, y1 = max(0, cy - rh // 2), min(H, cy + rh // 2)
        x0, x1 = max(0, cx - rw // 2), min(W, cx + rw // 2)
        region = img[y0:y1, x0:x1]
        
        hist, _ = np.histogram(region, bins=256, range=(0, 256))
        limite = max(1, int(CLIP_LIMIT * (region.size / 256)))
        exceso = np.maximum(hist - limite, 0).sum()
        hist = np.minimum(hist, limite) + (exceso // 256)
        hist[:exceso % 256] += 1
        
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
        denominador = region.size - cdf_min
        
        if denominador > 0:
            luts[r, c] = np.clip(np.round((cdf - cdf_min) / denominador * 255), 0, 255).astype(np.uint8)
        else:
            luts[r, c] = np.arange(256, dtype=np.uint8)


salida = np.zeros((H, W), dtype=np.uint8)

for y in range(H):
    r0 = int(np.clip(np.searchsorted(centros_y, y) - 1, 0, ny - 2))
    r1 = r0 + 1
    dy = np.clip((y - centros_y[r0]) / (centros_y[r1] - centros_y[r0] + 1e-10), 0.0, 1.0)

    for x in range(W):
        c0 = int(np.clip(np.searchsorted(centros_x, x) - 1, 0, nx - 2))
        c1 = c0 + 1
        dx = np.clip((x - centros_x[c0]) / (centros_x[c1] - centros_x[c0] + 1e-10), 0.0, 1.0)

        v = img[y, x]
        
        top = (1 - dx) * luts[r0, c0, v] + dx * luts[r0, c1, v]
        bot = (1 - dx) * luts[r1, c0, v] + dx * luts[r1, c1, v]
        salida[y, x] = np.round((1 - dy) * top + dy * bot)

cv2.imwrite("resultado_tarea.png", salida)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(salida, cmap='gray')
plt.title("Ecualización Local (Custom)")
plt.axis('off')

plt.tight_layout()
plt.show()
