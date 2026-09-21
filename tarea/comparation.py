import numpy as np
import matplotlib.pyplot as plt

# Reutilizamos tu función de interpolación periódica
def interpolacion_periodica_h(H, control_points):
    h_vals = np.array([p[0] for p in control_points])
    m_vals = np.array([p[1] for p in control_points])
    
    h_ext = np.concatenate([h_vals - 360.0, h_vals, h_vals + 360.0])
    m_ext = np.concatenate([m_vals, m_vals, m_vals])
    
    sort_idx = np.argsort(h_ext)
    h_ext = h_ext[sort_idx]
    m_ext = m_ext[sort_idx]
    
    return np.interp(H, h_ext, m_ext)

# Rango completo de tonos de 0° a 360°
h_rangos = np.linspace(0, 360, 500)

# Las 3 configuraciones
cfg1 = [(0.0, 1.0), (60.0, 2.5), (120.0, 1.0), (360.0, 1.0)]
cfg2 = [(0.0, 1.0), (180.0, 0.2), (240.0, 1.0), (360.0, 1.0)]
cfg3 = [(0.0, 1.0), (60.0, 2.0), (180.0, 0.3), (300.0, 1.5), (360.0, 1.0)]

# Calcular curvas
m_map1 = interpolacion_periodica_h(h_rangos, cfg1)
m_map2 = interpolacion_periodica_h(h_rangos, cfg2)
m_map3 = interpolacion_periodica_h(h_rangos, cfg3)

# Graficar
plt.figure(figsize=(10, 6))

plt.plot(h_rangos, m_map1, label='1. Aumento (Amarillos)', color='orange', linewidth=2)
plt.plot(h_rangos, m_map2, label='2. Atenuación (Cianes)', color='blue', linewidth=2)
plt.plot(h_rangos, m_map3, label='3. Mixta (Aumentos y Atenuaciones)', color='purple', linewidth=2, linestyle='--')

# Línea de referencia del valor neutro (m = 1.0)
plt.axhline(1.0, color='gray', linestyle=':', label='Valor Neutro (m = 1.0)')

plt.title('Comparación de Configuraciones de Puntos de Control m(h)')
plt.xlabel('Tono h (grados)')
plt.ylabel('Factor de Modificación m')
plt.xlim(0, 360)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('grafico_configuraciones_m.png', bbox_inches='tight', dpi=300)
print("¡Gráfico de configuraciones guardado exitosamente como 'grafico_configuraciones_m.png'!")