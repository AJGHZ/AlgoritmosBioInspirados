import numpy as np
import random

# PARAMETROS DEL AGORITMO
POPULATION_SIZE = 500 #Cantidad de tableros en cada generacion
MUTATION_RATE = 0.2 #Probabilidad de mutacion
ELITISM_COUNT = 50 #Cantidad de mejores individuos que se mantienen en la siguiente generacion
MAX_GENERATIONS = 1000 #Cantidad maxima de generaciones

print(f'Taza de mutación actual : {MUTATION_RATE}')

class SudokuGA:
    def __init__(self, puzzle_data):
        """
        Inicializa el problema con el tablero inicial.
        Puzzle: Lista de listas 9x9 con ceros en las celdas vacías y números del 1 al 9 en las celdas fijas.
        """
        self.grid = np.array(puzzle_data)
        self.fixed_mask = self.grid != 0 #Máscara para identificar celdas fijas
        self.history = [] #Para almacenar el error de la mejor solución en cada generación

    def fitness(self, individual):
        """
            FUNCIÓN OBJETIVO: Mide el error del tablero. Cuanto menor sea el error, mejor es la solución.
            Cada fila ya es valida por contrucción, asi que solo se cuentan los errores en columnas y bloques. El objetivo es minimizar el número de errores, idealmente llegando a 0.
        """
        errors = 0

        # Evaluar columnas: Falta de numeros únicos en cada columna
        for j in range(9):
            errors += (9 - len(np.unique(individual[:, j])))
            
        # Evaluar bloques 3x3: Falta de numeros únicos en cada bloque
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                block = individual[r:r+3, c:c+3]
                errors += (9 - len(np.unique(block)))

        return errors
        
    def create_individual(self):
        """
        Crea un tablero donde cada fila es una permutación válida de los números del 1 al 9, respetando las celdas fijas del puzzle original.
        """

        new_ind = self.grid.copy()
        for i in range(9):
            #identificar qué números faltan en la fila actual
            missing = [n for n in range(1,10) if n not in new_ind[i]]
            random.shuffle(missing) #mezclar los números faltantes para generar diversidad

             #Rellenar solo las posiciones que no son fijas (donde el valor es 0)
            for j in range(9):
                if not self.fixed_mask[i,j]: #si la posición no es fija
                    new_ind[i,j] = missing.pop() #asignar un número faltante
        return new_ind
        
    def crossover(self, parent1, parent2):
        """
        Cruce: Combina 2 padres para crear un hijo.
        Corta los tableros en un punto aleatorio y combina las filas
        """
        child = parent1.copy()
        point = random.randint(1, 8) #punto de corte entre filas
        child[point:] = parent2[point:] #combina las filas del segundo padre
        return child
        
    def mutate(self, individual):
        """
        Mutación: Intercambia 2 numeros (no fijos) dentro de una misma fila.
        Esto mantiene la integridad de la fila pero cambia las columnas y bloques.
        """
        row_idx = random.randint(0, 8) #selecciona una fila aleatoria
         #Obtener los indices de las celdas que se pueden modificar en esa fila
        mutable_indices = np.where(~self.fixed_mask[row_idx])[0]

        if len(mutable_indices) >= 2:
            idx1,idx2 = random.sample(list(mutable_indices),2)
            individual[row_idx, idx1], individual[row_idx, idx2] = \
                individual[row_idx, idx2], individual[row_idx, idx1] #intercambia los valores
                
    def solve(self):
        """Eejecuta el ciclo evolutivo para encontrar una solución al Sudoku."""
        # 1. Inicialización: Crear una población de tableros aleatorios respetando las celdas fijas.

        population = [self.create_individual() for _ in range(POPULATION_SIZE)]

        for gen in range(MAX_GENERATIONS):
            # 2. Evaluación: Ordenar pobación por fitness (menor error es mejor)
            population.sort(key = self.fitness)
            best_ind = population[0]
            current_error = self.fitness(best_ind)
            self.history.append(current_error)

            if gen % 50 == 0:
                print(f"Generación {gen:4} | Errores actuales: {current_error}")
            # Condición de parada: Si encontramos una solución perfecta (error = 0)
            if current_error == 0:
                print(f"¡Solución encontrada en la generación {gen}!")
                return best_ind
            # 3. Selección: Mantener los mejores individuos (elitismo)
            next_generation = population[:ELITISM_COUNT]
            # 4. Reproducción: Crear nuevos individuos mediante cruce y mutación
            while len(next_generation) < POPULATION_SIZE:
                parent1, parent2 = random.sample(population[:100], 2) # Selección de padres entre los mejores 100 individuos
                child = self.crossover(parent1, parent2) # Cruce
                if random.random() < MUTATION_RATE:
                    self.mutate(child) # Mutación
                next_generation.append(child)
            population = next_generation # Nueva generación
            
        print("No se encontró una solución perfecta en el número máximo de generaciones.")
        return population[0] # Devolver el mejor individuo encontrado
        
        # Datos de entrada (0 representa celdas vacías)

puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0 ,0 ,5],
        [0 ,0 ,0 ,0 ,8 ,0 ,0 ,7 ,9]
        ]

        # Ejecutar el algoritmo genético para resolver el Sudoku
solver = SudokuGA(puzzle)
solution = solver.solve()

print("\n--- Tablero final ---")
print(solution)