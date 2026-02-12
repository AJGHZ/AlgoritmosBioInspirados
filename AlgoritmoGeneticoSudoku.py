import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple

class SudokuGenetic:
    """Algoritmo genético para resolver Sudoku"""
    
    def __init__(self, sudoku_inicial: np.ndarray, tamanio_poblacion: int = 100):
        """
        Inicializa el algoritmo genético
        
        Args:
            sudoku_inicial: Matriz 9x9 con el sudoku parcialmente resuelto (0 = celda vacía)
            tamanio_poblacion: Tamaño de la población
        """
        self.sudoku_inicial = sudoku_inicial.copy()
        self.tamanio_poblacion = tamanio_poblacion
        self.tamanio_bloque = 3
        self.tamanio_tablero = 9
        
        # Identificar celdas fijas
        self.celdas_fijas = (sudoku_inicial != 0).copy()
        self.mejor_fitness = 0
        self.mejor_solucion = None
        self.historial_fitness = []
        
    def crear_individuo(self) -> np.ndarray:
        """Crea un individuo (solución de sudoku) válido aleatoriamente"""
        individuo = self.sudoku_inicial.copy()
        
        # Llenar celdas vacías con números aleatorios válidos
        for i in range(self.tamanio_tablero):
            for j in range(self.tamanio_tablero):
                if individuo[i, j] == 0:  # Celda vacía
                    # Obtener números posibles
                    numeros_posibles = self._obtener_numeros_validos(individuo, i, j)
                    if numeros_posibles:
                        individuo[i, j] = random.choice(numeros_posibles)
                    else:
                        # Si no hay números válidos, usar aleatorio y validar después
                        individuo[i, j] = random.randint(1, 9)
        
        return individuo
    
    def _obtener_numeros_validos(self, sudoku: np.ndarray, fila: int, col: int) -> List[int]:
        """Obtiene los números válidos para una celda"""
        numeros_usados = set()
        
        # Números en la fila
        numeros_usados.update(sudoku[fila, :])
        
        # Números en la columna
        numeros_usados.update(sudoku[:, col])
        
        # Números en el bloque 3x3
        bloque_fila = (fila // self.tamanio_bloque) * self.tamanio_bloque
        bloque_col = (col // self.tamanio_bloque) * self.tamanio_bloque
        numeros_usados.update(sudoku[bloque_fila:bloque_fila+3, bloque_col:bloque_col+3].flatten())
        
        # Retornar números disponibles
        return [n for n in range(1, 10) if n not in numeros_usados]
    
    def fitness(self, individuo: np.ndarray) -> int:
        """
        Calcula el fitness de un individuo
        Cuenta el número de celdas sin conflictos
        Mayor fitness = mejor solución
        """
        score = 0
        
        # Verificar filas
        for fila in range(self.tamanio_tablero):
            score += len(set(individuo[fila, :]))
        
        # Verificar columnas
        for col in range(self.tamanio_tablero):
            score += len(set(individuo[:, col]))
        
        # Verificar bloques 3x3
        for bloque_fila in range(0, self.tamanio_tablero, self.tamanio_bloque):
            for bloque_col in range(0, self.tamanio_tablero, self.tamanio_bloque):
                bloque = individuo[bloque_fila:bloque_fila+3, bloque_col:bloque_col+3].flatten()
                score += len(set(bloque))
        
        return score
    
    def seleccion_torneo(self, poblacion: List[np.ndarray], fitness_poblacion: List[int], tamanio_torneo: int = 3) -> np.ndarray:
        """Selecciona un individuo usando selección por torneo"""
        indices = random.sample(range(len(poblacion)), tamanio_torneo)
        mejor_indice = max(indices, key=lambda i: fitness_poblacion[i])
        return poblacion[mejor_indice].copy()
    
    def cruza_uniforme(self, padre1: np.ndarray, padre2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cruza uniforme: cada gen viene de uno u otro padre"""
        hijo1 = padre1.copy()
        hijo2 = padre2.copy()
        
        # Para cada celda no fija, intercambiar con probabilidad 0.5
        for i in range(self.tamanio_tablero):
            for j in range(self.tamanio_tablero):
                if not self.celdas_fijas[i, j]:
                    if random.random() < 0.5:
                        hijo1[i, j], hijo2[i, j] = hijo2[i, j], hijo1[i, j]
        
        return hijo1, hijo2
    
    def mutacion(self, individuo: np.ndarray, tasa_mutacion: float = 0.01) -> np.ndarray:
        """Mutación: intercambia valores en celdas no fijas"""
        mutante = individuo.copy()
        
        for i in range(self.tamanio_tablero):
            for j in range(self.tamanio_tablero):
                if not self.celdas_fijas[i, j] and random.random() < tasa_mutacion:
                    # Intercambiar con otra celda en la misma fila
                    otra_col = random.randint(0, 8)
                    if not self.celdas_fijas[i, otra_col]:
                        mutante[i, j], mutante[i, otra_col] = mutante[i, otra_col], mutante[i, j]
        
        return mutante
    
    def es_solucion(self, individuo: np.ndarray) -> bool:
        """Verifica si el individuo es una solución válida"""
        fitness_value = self.fitness(individuo)
        # Fitness máximo es 27 * 9 = 243 (9 filas + 9 columnas + 9 bloques, cada uno con suma 45)
        return fitness_value == 243
    
    def resolver(self, generaciones: int = 500, verbose: bool = True) -> Tuple[bool, np.ndarray, int]:
        """
        Ejecuta el algoritmo genético
        
        Args:
            generaciones: Número máximo de generaciones
            verbose: Mostrar progreso
            
        Returns:
            (solucion_encontrada, sudoku_resuelto, generación_encontrada)
        """
        # Crear población inicial
        poblacion = [self.crear_individuo() for _ in range(self.tamanio_poblacion)]
        
        for generacion in range(generaciones):
            # Calcular fitness de la población
            fitness_poblacion = [self.fitness(ind) for ind in poblacion]
            
            # Registrar mejor solución
            mejor_idx = np.argmax(fitness_poblacion)
            mejor_fitness_gen = fitness_poblacion[mejor_idx]
            
            if mejor_fitness_gen > self.mejor_fitness:
                self.mejor_fitness = mejor_fitness_gen
                self.mejor_solucion = poblacion[mejor_idx].copy()
            
            self.historial_fitness.append(self.mejor_fitness)
            
            if verbose and generacion % 50 == 0:
                print(f"Generación {generacion}: Fitness = {self.mejor_fitness}")
            
            # Verificar si se encontró solución
            if self.es_solucion(self.mejor_solucion):
                if verbose:
                    print(f"¡Solución encontrada en generación {generacion}!")
                return True, self.mejor_solucion, generacion
            
            # Crear nueva población
            nueva_poblacion = []
            
            # Elitismo: mantener al mejor individuo
            nueva_poblacion.append(self.mejor_solucion.copy())
            
            # Generar nueva población mediante cruza y mutación
            while len(nueva_poblacion) < self.tamanio_poblacion:
                # Seleccionar padres
                padre1 = self.seleccion_torneo(poblacion, fitness_poblacion)
                padre2 = self.seleccion_torneo(poblacion, fitness_poblacion)
                
                # Cruza
                hijo1, hijo2 = self.cruza_uniforme(padre1, padre2)
                
                # Mutación
                hijo1 = self.mutacion(hijo1)
                if len(nueva_poblacion) < self.tamanio_poblacion:
                    hijo2 = self.mutacion(hijo2)
                
                nueva_poblacion.append(hijo1)
                if len(nueva_poblacion) < self.tamanio_poblacion:
                    nueva_poblacion.append(hijo2)
            
            poblacion = nueva_poblacion[:self.tamanio_poblacion]
        
        if verbose:
            print(f"No se encontró solución. Mejor fitness: {self.mejor_fitness}")
        
        return False, self.mejor_solucion, generaciones


def imprimir_sudoku(sudoku: np.ndarray) -> None:
    """Imprime el sudoku de forma legible"""
    print("\n" + "="*25)
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 25)
        fila = ""
        for j in range(9):
            if j % 3 == 0 and j != 0:
                fila += "| "
            fila += str(int(sudoku[i, j])) + " "
        print(fila)
    print("="*25 + "\n")


# Ejemplo de uso
if __name__ == "__main__":
    # Sudoku de ejemplo (0 = celda vacía)
    sudoku_ejemplo = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    print("SUDOKU INICIAL:")
    imprimir_sudoku(sudoku_ejemplo)
    
    # Crear y ejecutar el algoritmo genético
    print("Resolviendo con Algoritmo Genético...")
    ag = SudokuGenetic(sudoku_ejemplo, tamanio_poblacion=150)
    encontrado, solucion, gen = ag.resolver(generaciones=1000, verbose=True)
    
    if encontrado:
        print("\nSUDOKU RESUELTO:")
        imprimir_sudoku(solucion)
    else:
        print("\nMejor solución obtenida:")
        imprimir_sudoku(ag.mejor_solucion)
        print(f"Fitness: {ag.mejor_fitness}/243")
