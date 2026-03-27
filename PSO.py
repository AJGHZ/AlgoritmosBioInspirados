import numpy as np

# Configuración del algoritmo
n_particulas = 30
dimensiones = 2
iteraciones = 50
w = 0.5  # Inercia
c1 = 1.5 # Coeficiente cognitivo (propio)
c2 = 2.0 # Coeficiente social (enjambre)

# Función objetivo (lo que queremos minimizar)
def funcion_objetivo(x):
    return np.sum(x**2)

# Inicialización
posiciones = np.random.uniform(-5, 5, (n_particulas, dimensiones))
velocidades = np.random.uniform(-1, 1, (n_particulas, dimensiones))

pbest_pos = posiciones.copy()
pbest_valor = np.array([funcion_objetivo(p) for p in posiciones])

gbest_pos = pbest_pos[np.argmin(pbest_valor)]
gbest_valor = np.min(pbest_valor)

# Ciclo de optimización
for t in range(iteraciones):
    for i in range(n_particulas):
        # 1. Actualizar velocidad
        r1, r2 = np.random.rand(2)
        velocidades[i] = (w * velocidades[i] + 
                          c1 * r1 * (pbest_pos[i] - posiciones[i]) + 
                          c2 * r2 * (gbest_pos - posiciones[i]))
        
        # 2. Actualizar posición
        posiciones[i] += velocidades[i]
        
        # 3. Evaluar y actualizar mejores marcas
        valor_actual = funcion_objetivo(posiciones[i])
        
        if valor_actual < pbest_valor[i]:
            pbest_valor[i] = valor_actual
            pbest_pos[i] = posiciones[i].copy()
            
            if valor_actual < gbest_valor:
                gbest_valor = valor_actual
                gbest_pos = posiciones[i].copy()

    if t % 10 == 0:
        print(f"Iteración {t}: Mejor valor = {gbest_valor:.5f}")

print(f"\nResultado final:")
print(f"Mejor posición: {gbest_pos}")
print(f"Valor mínimo: {gbest_valor:.10f}")