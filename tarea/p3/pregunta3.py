import numpy as np
from skimage import io, color, img_as_ubyte

def reescalar_imagen(img, s, modo='bilineal'):
    """
    Reescala una imagen (grises o RGB) por un factor real s en [0.5, 2.0].
    Modos: 'vecino' (Vecino más cercano) o 'bilineal' (Interpolación bilineal).
    """
    if not (0.5 <= s <= 2.0):
        raise ValueError("El factor s debe estar estrictamente en el intervalo [0.5, 2.0]")
    
    # Determinar si es RGB o escala de grises
    is_rgb = img.ndim == 3
    if is_rgb:
        H_in, W_in, C = img.shape
    else:
        H_in, W_in = img.shape
        C = 1

    # Cálculo automático del tamaño de salida (redondeo al entero más cercano)
    H_out = int(np.round(H_in * s))
    W_out = int(np.round(W_in * s))
    
    # Inicializar arreglo de salida
    if is_rgb:
        salida = np.zeros((H_out, W_out, C), dtype=np.float64)
    else:
        salida = np.zeros((H_out, W_out), dtype=np.float64)

    # Convención de coordenadas (mapeo inverso: i_in = i_out / s)
    for i_out in range(H_out):
        for j_out in range(W_out):
            r_in = i_out / s
            c_in = j_out / s

            if modo == 'vecino':
                # Vecino más cercano
                r_idx = int(np.round(r_in))
                c_idx = int(np.round(c_in))
                
                # Tratamiento de bordes mediante recorte de índices
                r_idx = np.clip(r_idx, 0, H_in - 1)
                c_idx = np.clip(c_idx, 0, W_in - 1)
                
                if is_rgb:
                    salida[i_out, j_out, :] = img[r_idx, c_idx, :]
                else:
                    salida[i_out, j_out] = img[r_idx, c_idx]

            elif modo == 'bilineal':
                # Interpolación bilineal: cálculo de los 4 vecinos enteros envolventes
                r0 = int(np.floor(r_in))
                c0 = int(np.floor(c_in))
                r1 = r0 + 1
                c1 = c0 + 1

                # Índices seguros con restricciones de borde
                r0_safe = max(0, min(r0, H_in - 1))
                r1_safe = max(0, min(r1, H_in - 1))
                c0_safe = max(0, min(c0, W_in - 1))
                c1_safe = max(0, min(c1, W_in - 1))

                # Cálculo de pesos basados en las distancias fraccionales
                # Si r0 == r1 o c0 == c1 (en los bordes exactos), manejamos las ponderaciones
                w00 = (r1 - r_in) * (c1 - c_in)
                w01 = (r1 - r_in) * (c_in - c0)
                w10 = (r_in - r0) * (c1 - c_in)
                w11 = (r_in - r0) * (c_in - c0)

                suma_pesos = w00 + w01 + w10 + w11
                if suma_pesos > 0:
                    w00 /= suma_pesos
                    w01 /= suma_pesos
                    w10 /= suma_pesos
                    w11 /= suma_pesos

                if is_rgb:
                    val = (w00 * img[r0_safe, c0_safe, :] +
                           w01 * img[r0_safe, c1_safe, :] +
                           w10 * img[r1_safe, c0_safe, :] +
                           w11 * img[r1_safe, c1_safe, :])
                    salida[i_out, j_out, :] = val
                else:
                    val = (w00 * img[r0_safe, c0_safe] +
                           w01 * img[r0_safe, c1_safe] +
                           w10 * img[r1_safe, c0_safe] +
                           w11 * img[r1_safe, c1_safe])
                    salida[i_out, j_out] = val

    # Retornar en el tipo de datos adecuado
    if img.dtype == np.uint8 or np.max(img) > 1:
        return np.clip(salida, 0, 255).astype(np.uint8)
    else:
        return np.clip(salida, 0.0, 1.0)

if __name__ == '__main__':
    # Ejemplo de uso: Cargar imagen y reescalar con factor de ampliación s = 1.5
    imagen = io.imread('P3_IMG_2387_crop.tif')
    resultado_bilineal = reescalar_imagen(imagen, s=1.5, modo='bilineal')
    io.imsave('resultado_p3_ampliado.png', resultado_bilineal)
    print("¡Reescalado completado y guardado con éxito!")