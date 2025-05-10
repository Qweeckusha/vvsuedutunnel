import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, value


# ========== ФУНКЦИИ ДЛЯ РЕШЕНИЯ СУДОКУ ==========

def solve_single_sudoku(grid):
    """ Решает одиночное судоку с использованием PuLP """
    model = LpProblem("Sudoku_Solver", LpMinimize)

    # Создание переменных (7)
    x = [[[LpVariable(f"x_{i}_{j}_{k}", cat=LpBinary) for k in range(9)]
          for j in range(9)] for i in range(9)]

    # Целевая функция (2)
    model += 0

    # Ограничения для ячеек - одно значение в ячейке (5)
    for i in range(9):
        for j in range(9):
            model += lpSum(x[i][j][k] for k in range(9)) == 1  # (5)

    # Ограничения для строк - уникальность в строке (3)
    for i in range(9):
        for k in range(9):
            model += lpSum(x[i][j][k] for j in range(9)) == 1  # (3)

    # Ограничения для столбцов - уникальность в столбце (4)
    for j in range(9):
        for k in range(9):
            model += lpSum(x[i][j][k] for i in range(9)) == 1  # (4)

    # Ограничения для блоков - уникальность в блоке 3x3 (6)
    for block_row in range(3):
        for block_col in range(3):
            for k in range(9):
                model += lpSum(x[i][j][k]
                               for i in range(3 * block_row, 3 * block_row + 3)
                               for j in range(3 * block_col, 3 * block_col + 3)) == 1  # (6)

    # Предзаполненные ячейки (8)
    for i in range(9):
        for j in range(9):
            num = grid[i][j]
            if num != 0:
                model += x[i][j][num - 1] == 1  # (8)

    model.solve()

    if model.status == 1:
        solution = [[0] * 9 for _ in range(9)]
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if value(x[i][j][k]) > 0.5:
                        solution[i][j] = k + 1
        return solution
    else:
        print("Решение не найдено")
        return None


def solve_multiple_sudokus(sudokus, overlaps=None):
    """ Решает несколько судоку с пересекающимися областями """
    num_sudokus = len(sudokus)
    model = LpProblem("Multiple_Sudoku_Solver", LpMinimize)

    # Создание переменных (14)
    x = [[[[LpVariable(f"x_{n}_{i}_{j}_{k}", cat=LpBinary) for k in range(9)]
           for j in range(9)] for i in range(9)] for n in range(num_sudokus)]

    # Целевая функция (9)
    model += 0

    # Ограничения для ячеек - одно значение в ячейке (12)
    for n in range(num_sudokus):
        for i in range(9):
            for j in range(9):
                model += lpSum(x[n][i][j][k] for k in range(9)) == 1  # (12)

    # Ограничения для строк - уникальность в строке (10)
    for n in range(num_sudokus):
        for j in range(9):
            for k in range(9):
                model += lpSum(x[n][i][j][k] for i in range(9)) == 1  # (10)

    # Ограничения для столбцов - уникальность в столбце (11)
    for n in range(num_sudokus):
        for i in range(9):
            for k in range(9):
                model += lpSum(x[n][i][j][k] for j in range(9)) == 1  # (11)

    # Ограничения для блоков - уникальность в блоке 3x3 (13)
    for n in range(num_sudokus):
        for block_row in range(3):
            for block_col in range(3):
                for k in range(9):
                    model += lpSum(x[n][i][j][k]
                                   for i in range(3 * block_row, 3 * block_row + 3)
                                   for j in range(3 * block_col, 3 * block_col + 3)) == 1  # (13)

    # Предзаполненные ячейки (15)
    for n in range(num_sudokus):
        grid = sudokus[n]
        for i in range(9):
            for j in range(9):
                num = grid[i][j]
                if num != 0:
                    model += x[n][i][j][num - 1] == 1  # (15)

    # Ограничения для пересекающихся областей (16)
    if overlaps:
        for ((n1, i1, j1), (n2, i2, j2)) in overlaps:
            # Только для ячеек в пересекающихся областях (все нули в оригинале)
            if sudokus[n1][i1][j1] == 0 and sudokus[n2][i2][j2] == 0:
                for k in range(9):
                    model += x[n1][i1][j1][k] == x[n2][i2][j2][k]  # (16)

    model.solve()

    if model.status == 1:
        solutions = []
        for n in range(num_sudokus):
            solution = [[0] * 9 for _ in range(9)]
            for i in range(9):
                for j in range(9):
                    for k in range(9):
                        if value(x[n][i][j][k]) > 0.5:
                            solution[i][j] = k + 1
            solutions.append(solution)
        return solutions
    else:
        print("Решение не найдено")
        return None


# Обновленное определение пересечений для тройного судоку
overlaps_triple = [
    # Пересечение первого и второго судоку (правый нижний угол первого и левый верхний второго)
    *[((0, i + 6, j + 6), (1, i, j)) for i in range(3) for j in range(3)],
    # Пересечение второго и третьего судоку (левый нижний угол второго и правый верхний третьего)
    *[((1, i + 6, j), (2, i, j + 6)) for i in range(3) for j in range(3)]
]


# ========== ФУНКЦИИ ВИЗУАЛИЗАЦИИ ==========

def visualize_sudoku(grid, title="Решение судоку"):
    """ Визуализирует одно судоку """
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ListedColormap(["white", "lightgray"])

    # Создание фона для блоков
    block_mask = np.zeros((9, 9), dtype=int)
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            block_mask[i:i + 3, j:j + 3] = 1

    ax.imshow(block_mask, cmap=cmap, extent=[0, 9, 0, 9])

    # Добавление чисел
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                ax.text(j + 0.5, 8.5 - i, str(grid[i][j]),
                        ha='center', va='center', fontsize=14)

    # Добавление линий сетки
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axvline(i, color='black', linewidth=lw)
        ax.axhline(i, color='black', linewidth=lw)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.show()


def visualize_combined_double(sud1, sud2, offset_x=3, offset_y=3):
    """ Визуализирует два судоку с пересекающейся областью """
    combined = np.zeros((15, 21), dtype=int)

    # Размещение первого судоку
    for i in range(9):
        for j in range(9):
            combined[i][j] = sud1[i][j]

    # Размещение второго судоку со смещением
    for i in range(9):
        for j in range(9):
            combined[i + offset_x][j + offset_y] = sud2[i][j]

    # Проверка конфликтов в пересечении
    for i in range(9):
        for j in range(9):
            if 0 <= i + offset_x < 15 and 0 <= j + offset_y < 21:
                val1 = combined[i + offset_x][j + offset_y]
                val2 = sud2[i][j]
                if val1 != 0 and val2 != 0 and val1 != val2:
                    print(f"Конфликт в ({i + offset_x}, {j + offset_y}): {val1} vs {val2}")

    # Визуализация
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = ListedColormap(["white", "lightgray"])

    # Создание фона для блоков
    block_mask = np.zeros((15, 21), dtype=int)
    for i in range(0, 15, 3):
        for j in range(0, 21, 3):
            block_mask[i:i + 3, j:j + 3] = 1

    ax.imshow(block_mask, cmap=cmap, extent=[0, 21, 0, 15])

    # Добавление чисел
    for i in range(15):
        for j in range(21):
            if combined[i][j] != 0:
                ax.text(j + 0.5, 14.5 - i, str(combined[i][j]),
                        ha='center', va='center', fontsize=14)

    # Добавление линий сетки
    for i in range(22):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axvline(i, color='black', linewidth=lw)
    for i in range(16):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i, color='black', linewidth=lw)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Двойное судоку с пересечением")
    plt.show()


def visualize_combined_triple(sud1, sud2, sud3):
    """ Визуализирует три судоку с пересекающимися областями """
    combined = np.zeros((15, 21), dtype=int)

    # Размещение первого судоку в левом верхнем углу
    for i in range(9):
        for j in range(9):
            combined[i][j] = sud1[i][j]

    # Размещение второго судоку справа (пересекается с первым)
    for i in range(9):
        for j in range(9):
            combined[i][j + 12] = sud2[i][j]

    # Размещение третьего судоку снизу (пересекается со вторым)
    for i in range(9):
        for j in range(9):
            combined[i + 6][j + 6] = sud3[i][j]

    # Визуализация
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = ListedColormap(["white", "lightgray"])

    # Создание фона для блоков
    block_mask = np.zeros((15, 21), dtype=int)
    for i in range(0, 15, 3):
        for j in range(0, 21, 3):
            block_mask[i:i + 3, j:j + 3] = 1

    ax.imshow(block_mask, cmap=cmap, extent=[0, 21, 0, 15])

    # Добавление чисел
    for i in range(15):
        for j in range(21):
            if combined[i][j] != 0:
                ax.text(j + 0.5, 14.5 - i, str(combined[i][j]),
                        ha='center', va='center', fontsize=14)

    # Добавление линий сетки
    for i in range(22):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axvline(i, color='black', linewidth=lw)
    for i in range(16):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i, color='black', linewidth=lw)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Тройное судоку с пересечениями")
    plt.show()


# ========== МАТРИЦЫ СУДОКУ ==========

# Одиночные судоку
initial_grid_easy = [
    [2, 0, 5, 0, 0, 9, 0, 0, 4],
    [0, 0, 0, 0, 0, 0, 3, 0, 7],
    [7, 0, 0, 8, 5, 6, 0, 1, 0],
    [4, 5, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 8, 5],
    [0, 2, 0, 4, 1, 8, 0, 0, 6],
    [6, 0, 8, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 2, 0, 0, 7, 0, 8]
]

initial_grid_medium = [
    [0, 0, 6, 0, 9, 0, 2, 0, 0],
    [0, 0, 0, 7, 0, 2, 0, 0, 0],
    [0, 9, 0, 5, 0, 8, 0, 7, 0],
    [9, 0, 0, 0, 3, 0, 0, 0, 6],
    [7, 5, 0, 0, 0, 0, 0, 1, 9],
    [1, 0, 0, 0, 4, 0, 0, 0, 5],
    [0, 1, 0, 3, 0, 9, 0, 8, 0],
    [0, 0, 0, 2, 0, 1, 0, 0, 0],
    [0, 0, 9, 0, 8, 0, 1, 0, 0]
]

initial_grid_hard = [
    [0, 0, 0, 8, 0, 0, 0, 0, 0],
    [7, 8, 9, 0, 1, 0, 0, 0, 6],
    [0, 0, 0, 0, 0, 6, 1, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 5, 0],
    [5, 0, 8, 7, 0, 9, 3, 0, 4],
    [0, 4, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 3, 2, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 7, 0, 4, 3, 9],
    [0, 0, 0, 0, 0, 1, 0, 0, 0]
]

# Двойные судоку - Набор 1
double_grid_1_1 = [
    [0, 0, 0, 0, 0, 2, 5, 0, 6],
    [7, 1, 0, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 9, 0, 0, 0, 4, 0, 0],
    [0, 7, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 8, 0, 0, 0, 3],
    [0, 0, 0, 2, 0, 0, 0, 5, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 2, 0, 0, 9, 0, 0, 0]
]

double_grid_1_2 = [
    [0, 0, 0, 4, 0, 0, 8, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 8, 0, 0, 0, 3, 0, 0, 0],
    [2, 0, 0, 0, 5, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 9, 0, 0, 0, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 0, 0, 2, 9],
    [8, 0, 7, 3, 0, 0, 0, 0, 0]
]

# Двойные судоку - Набор 2
double_grid_2_1 = [
    [4, 0, 8, 3, 0, 0, 0, 0, 2],
    [7, 6, 0, 0, 0, 0, 0, 9, 0],
    [0, 0, 5, 0, 0, 0, 8, 7, 1],
    [0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 5, 0, 9, 7, 4, 0, 6, 0],
    [0, 7, 0, 0, 0, 6, 0, 0, 0],
    [0, 4, 1, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 1, 0, 4],
    [5, 0, 0, 0, 0, 1, 9, 0, 0]
]

double_grid_2_2 = [
    [0, 0, 0, 6, 9, 0, 0, 4, 2],
    [1, 0, 4, 8, 0, 0, 0, 0, 0],
    [9, 0, 0, 0, 0, 0, 8, 0, 1],
    [8, 6, 0, 2, 0, 7, 0, 0, 4],
    [0, 0, 0, 3, 8, 6, 0, 0, 0],
    [3, 0, 0, 9, 4, 5, 0, 2, 8],
    [2, 0, 8, 0, 0, 0, 1, 0, 5],
    [7, 0, 0, 0, 0, 0, 4, 0, 3],
    [6, 1, 0, 0, 5, 9, 0, 0, 0]
]

# Тройные судоку
triple_grid_3_1 = [
    [6, 0, 8, 0, 1, 0, 0, 0, 0],
    [0, 0, 7, 0, 4, 0, 8, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 9],
    [0, 6, 0, 7, 0, 0, 0, 3, 0],
    [0, 7, 0, 5, 0, 4, 0, 0, 0],
    [0, 0, 0, 8, 0, 0, 9, 5, 0],
    [0, 0, 0, 0, 7, 8, 0, 0, 0],
    [5, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 4, 0, 0, 6, 0, 0, 0]
]

triple_grid_3_2 = [
    [0, 0, 0, 0, 6, 0, 3, 0, 9],
    [7, 0, 3, 0, 4, 0, 1, 0, 0],
    [6, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 0, 6, 0, 9, 0],
    [0, 0, 0, 7, 0, 4, 0, 8, 0],
    [0, 5, 4, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 4, 9, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 9, 0, 5],
    [0, 0, 0, 3, 0, 0, 4, 0, 0]
]

triple_grid_3_3 = [
    [0, 0, 0, 1, 0, 8, 0, 0, 0],
    [0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 7, 0, 0, 0, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 6],
    [0, 0, 0, 6, 5, 2, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 5, 0, 3, 0, 0, 0],
    [0, 8, 7, 0, 0, 0, 6, 5, 0],
    [0, 4, 0, 0, 0, 0, 0, 8, 0]
]

# ========== ОПРЕДЕЛЕНИЯ ПЕРЕСЕЧЕНИЙ ==========

# Пересечения для двойного судоку - Набор 1 (4 пересекающихся блока)
overlaps_double_1 = [((0, i + 3, j + 3), (1, i, j)) for i in range(6) for j in range(6)]

# Пересечения для двойного судоку - Набор 2 (1 пересекающийся блок)
overlaps_double_2 = [((0, i + 6, j + 6), (1, i, j)) for i in range(3) for j in range(3)]

# Пересечения для тройного судоку
overlaps_triple = [
    # Пересечение первого и второго судоку (правый нижний угол первого и левый верхний второго)
    *[((0, i + 6, j + 6), (1, i, j)) for i in range(3) for j in range(3)],
    # Пересечение второго и третьего судоку (левый нижний угол второго и правый верхний третьего)
    *[((1, i + 6, j), (2, i, j + 6)) for i in range(3) for j in range(3)]
]

# ========== ОСНОВНОЙ КОД ==========

if __name__ == "__main__":
    print("=== Решение одиночных судоку ===")
    easy_solution = solve_single_sudoku(initial_grid_easy)
    if easy_solution:
        visualize_sudoku(easy_solution, "Легкое судоку - решение")

    medium_solution = solve_single_sudoku(initial_grid_medium)
    if medium_solution:
        visualize_sudoku(medium_solution, "Среднее судоку - решение")

    hard_solution = solve_single_sudoku(initial_grid_hard)
    if hard_solution:
        visualize_sudoku(hard_solution, "Сложное судоку - решение")

    print("\n=== Решение двойного судоку - Набор 1 ===")
    double_solutions_1 = solve_multiple_sudokus([double_grid_1_1, double_grid_1_2], overlaps_double_1)
    if double_solutions_1:
        visualize_combined_double(*double_solutions_1, offset_x=3, offset_y=3)

    print("\n=== Решение двойного судоку - Набор 2 ===")
    double_solutions_2 = solve_multiple_sudokus([double_grid_2_1, double_grid_2_2], overlaps_double_2)
    if double_solutions_2:
        visualize_combined_double(*double_solutions_2, offset_x=6, offset_y=6)

    print("\n=== Решение тройного судоку ===")
    triple_solutions = solve_multiple_sudokus([triple_grid_3_1, triple_grid_3_2, triple_grid_3_3], overlaps_triple)
    if triple_solutions:
        visualize_combined_triple(*triple_solutions)