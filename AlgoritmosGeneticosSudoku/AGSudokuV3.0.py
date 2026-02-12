import numpy as np
import random

# PARAMETROS DEL ALGORITMO
POPULATION_SIZE = 700 # Subimos un poco para dar más variedad
MUTATION_RATE = 0.5   # Empezamos alto para explorar agresivamente
ELITISM_COUNT = 50 
MAX_GENERATIONS = 1500 # Damos más margen para las sacudidas de diversidad

print('\nAlgoritomo de Sudoku Genético - Versión 3.0')
print(f'Taza de mutación actual : {MUTATION_RATE}\n')

class SudokuGA:
    def __init__(self, puzzle_data):
        self.grid = np.array(puzzle_data)
        self.fixed_mask = self.grid != 0 
        self.history = [] 

    def fitness(self, individual):
        errors = 0
        # Columnas
        for j in range(9):
            errors += (9 - len(np.unique(individual[:, j])))
        # Bloques
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                block = individual[r:r+3, c:c+3]
                errors += (9 - len(np.unique(block)))
        return errors
        
    def create_individual(self):
        new_ind = self.grid.copy()
        for i in range(9):
            missing = [n for n in range(1,10) if n not in new_ind[i]]
            random.shuffle(missing)
            for j in range(9):
                if not self.fixed_mask[i,j]:
                    new_ind[i,j] = missing.pop()
        return new_ind
        
    def crossover(self, parent1, parent2):
        child = parent1.copy()
        point = random.randint(1, 8)
        child[point:] = parent2[point:]
        return child
        
    def mutate(self, individual, current_error):
        """Mutación adaptativa: ajusta su fuerza según el error actual."""
        # Si el error es menor a 5, usamos mutación quirúrgica (0.1)
        # Si es mayor, usamos la tasa agresiva (MUTATION_RATE)
        prob = 0.1 if current_error < 5 else MUTATION_RATE
        
        if random.random() < prob:
            row_idx = random.randint(0, 8)
            mutable_indices = np.where(~self.fixed_mask[row_idx])[0]
            if len(mutable_indices) >= 2:
                idx1, idx2 = random.sample(list(mutable_indices), 2)
                individual[row_idx, idx1], individual[row_idx, idx2] = \
                    individual[row_idx, idx2], individual[row_idx, idx1]
                
    def solve(self):
        population = [self.create_individual() for _ in range(POPULATION_SIZE)]
        
        best_overall_error = 100
        stagnation_counter = 0

        for gen in range(MAX_GENERATIONS):
            population.sort(key=self.fitness)
            best_ind = population[0]
            current_error = self.fitness(best_ind)
            self.history.append(current_error)

            # --- Lógica de Estancamiento ---
            if current_error < best_overall_error:
                best_overall_error = current_error
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if gen % 50 == 0:
                print(f"Gen {gen:4} | Error: {current_error} | Estancado: {stagnation_counter}")

            if current_error == 0:
                print(f"¡SOLUCIÓN ENCONTRADA EN LA GENERACIÓN {gen}!")
                return best_ind

            # --- Sacudida de Diversidad ---
            # Si llevamos 150 generaciones sin mejorar el récord personal
            if stagnation_counter > 150:
                print("--- SACUDIDA: Inyectando sangre nueva para salir del bache ---")
                # Mantenemos a los mejores 10, el resto son nuevos
                population[10:] = [self.create_individual() for _ in range(POPULATION_SIZE - 10)]
                stagnation_counter = 0

            # --- Evolución estándar ---
            next_generation = population[:ELITISM_COUNT]
            while len(next_generation) < POPULATION_SIZE:
                # Selección de padres (entre los mejores 150 para mayor presión)
                parent1, parent2 = random.sample(population[:150], 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child, current_error)
                next_generation.append(child)
            population = next_generation
            
        print("Fin del tiempo. Mejor resultado alcanzado:", best_overall_error)
        return population[0]

# --- DATOS Y EJECUCIÓN ---
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

solver = SudokuGA(puzzle)
solution = solver.solve()

print("\n--- Tablero final ---")
print(solution)