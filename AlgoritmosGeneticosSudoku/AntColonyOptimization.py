import numpy as np

class AntColonyOptimizer:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
        """
        Configuración del optimizador.
        :param distances: Matriz cuadrada donde [i][j] es la distancia de ciudad i a j.
        :param n_ants: Cantidad de hormigas que exploran en cada iteración.
        :param n_best: Número de mejores hormigas que tienen permiso para depositar feromona.
        :param n_iterations: Cuántas veces se repite el proceso de búsqueda.
        :param decay: Factor de persistencia de feromona (1 - tasa de evaporación).
        :param alpha: Peso relativo de la feromona (memoria colectiva).
        :param beta: Peso relativo de la visibilidad (distancia inmediata).
        """
        self.distances  = distances
        # Inicializamos la matriz de feromonas con valores pequeños iguales (0.1)
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        best_path = None
        all_time_best_path = ("Ruta", np.inf)
        
        for i in range(self.n_iterations):
            # 1. Generar rutas para todas las hormigas en esta iteración
            all_paths = self.gen_all_paths()
            
            # 2. Esparcir feromonas basado en los resultados obtenidos
            self.spread_pheromone(all_paths, self.n_best)
            
            # 3. Encontrar la mejor ruta de la iteración actual
            best_path = min(all_paths, key=lambda x: x[1])
            
            # 4. Actualizar el récord histórico si encontramos algo mejor
            if best_path[1] < all_time_best_path[1]:
                all_time_best_path = best_path            
            
            # 5. Evaporación: La feromona disminuye gradualmente en todo el mapa
            self.pheromone = self.pheromone * self.decay
            
        return all_time_best_path

    def spread_pheromone(self, all_paths, n_best):
        """Refuerza los caminos tomados por las mejores hormigas."""
        # Ordenamos las rutas de menor a mayor distancia
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                # La cantidad de feromona añadida es inversamente proporcional a la distancia
                # A camino más corto, rastro más fuerte.
                self.pheromone[move] += 1.0 / dist

    def gen_path(self, start):
        """Simula el recorrido de una sola hormiga."""
        path = []
        visited = set()
        visited.add(start)
        prev = start
        
        for i in range(len(self.distances) - 1):
            # Elegir el siguiente nodo basándose en probabilidad
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
            
        # Regresar a la ciudad de origen para cerrar el ciclo del TSP
        path.append((prev, start)) 
        return path

    def pick_move(self, pheromone, dist, visited):
        """Aplica la fórmula probabilística para decidir el siguiente paso."""
        pheromone = np.copy(pheromone)
        # No podemos volver a ciudades ya visitadas en este viaje (probabilidad 0)
        pheromone[list(visited)] = 0

        # Numerador de la fórmula: (Feromona^alpha) * ((1/Distancia)^beta)
        # Usamos 1e-10 para evitar división por cero si la distancia es 0
        row = (pheromone ** self.alpha) * ((1.0 / (dist + 1e-10)) ** self.beta)
        
        # Normalizamos para que la suma de probabilidades sea 1
        norm_row = row / row.sum()
        
        # Selección aleatoria basada en los pesos calculados
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

    def gen_all_paths(self):
        """Genera las rutas de todas las hormigas partiendo de la ciudad 0."""
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.calculate_total_dist(path)))
        return all_paths

    def calculate_total_dist(self, path):
        """Suma las distancias de cada segmento del camino."""
        total_dist = 0
        for segment in path:
            total_dist += self.distances[segment]
        return total_dist

# --- Ejemplo de aplicación ---

# Matriz de distancias (ejemplo de 5 ciudades)
# np.inf representa que no hay conexión directa o es el mismo nodo
dist_matrix = np.array([
    [np.inf, 10, 15, 20, 25],
    [10, np.inf, 35, 25, 12],
    [15, 35, np.inf, 30, 10],
    [20, 25, 30, np.inf,  5],
    [25, 12, 10,  5, np.inf]
])

# Instanciar y ejecutar
aco = AntColonyOptimizer(
    distances=dist_matrix, 
    n_ants=15, 
    n_best=5, 
    n_iterations=50, 
    decay=0.95, 
    alpha=1, 
    beta=2
)

mejor_ruta, distancia_minima = aco.run()

print(f"Resultado final:")
print(f" -> Ruta (segmentos i,j): {mejor_ruta}")
print(f" -> Distancia total: {distancia_minima}")