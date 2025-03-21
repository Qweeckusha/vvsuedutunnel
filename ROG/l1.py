import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# Функция для генерации транспортной задачи
def generate_transport_problem(m, n, seed=42):
    """
    Генерирует случайные данные для транспортной задачи.
    m - Количество пунктов производства (источников).
    n - Количество пунктов потребления (стоков).
    seed - Сид для воспроизводимости случайных чисел.
    return: supply (массив запасов), demand (массив спроса), cost (матрица стоимостей перевозки).
    """
    np.random.seed(seed)
    supply = np.random.randint(10, 50, size=m)  # Генерируем случайные запасы для каждого источника
    demand = np.random.randint(10, 50, size=n)  # Генерируем случайный спрос для каждого стока
    cost = np.random.randint(1, 20, size=(m, n))  # Генерируем матрицу стоимостей перевозки
    return supply, demand, cost


# Функция для решения транспортной задачи методом линейного программирования
def solve_transportation(supply, demand, cost):
    """
    Решает транспортную задачу методом линейного программирования.
    return: Оптимальное решение в виде матрицы перевозок или None, если решение не найдено.
    """
    m, n = len(supply), len(demand)  # Получаем количество источников и стоков
    c = cost.flatten()  # Преобразуем матрицу стоимостей в одномерный массив
    A_eq = []  # Матрица ограничений равенства
    b_eq = []  # Вектор правых частей ограничений равенства

    # Добавляем ограничения на выполнение всех поставок из каждого источника
    for i in range(m):
        row = [0] * (m * n)  # Создаем строку нулей
        for j in range(n):
            row[i * n + j] = 1  # Устанавливаем единицы для соответствующих переменных
        A_eq.append(row)
        b_eq.append(supply[i])  # Добавляем ограничение на общий объем поставок из источника

    # Добавляем ограничения на удовлетворение спроса каждого стока
    for j in range(n):
        row = [0] * (m * n)  # Создаем строку нулей
        for i in range(m):
            row[i * n + j] = 1  # Устанавливаем единицы для соответствующих переменных
        A_eq.append(row)
        b_eq.append(demand[j])  # Добавляем ограничение на общий объем спроса стока

    x_bounds = [(0, None)] * (m * n)  # Переменные должны быть неотрицательными
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')  # Решаем задачу ЛП
    return result.x.reshape(m, n) if result.success else None  # Возвращаем решение в виде матрицы


# Функция для визуализации графа транспортной задачи
def draw_graph(supply, demand, cost, solution=None):
    """
    Выводим граф транспортной задачи.
    solution: Оптимальное решение (матрица перевозок).
    """
    G = nx.DiGraph()  # Создаем ориентированный граф
    m, n = len(supply), len(demand)  # Получаем количество источников и стоков

    # Добавляем узлы для источников
    for i in range(m):
        G.add_node(f"S{i}", demand=-supply[i])

    # Добавляем узлы для стоков
    for j in range(n):
        G.add_node(f"D{j}", demand=demand[j])


    for i in range(m):
        for j in range(n):
            weight = cost[i][j]  # Стоимость перевозки между источником и стоком
            G.add_edge(f"S{i}", f"D{j}", weight=weight, label=f"{weight}")

    # Определяем позиции узлов для визуализации
    pos = {f"S{i}": (0, -i) for i in range(m)}  # Источники слева
    pos.update({f"D{j}": (1, -j) for j in range(n)})  # Стоки справа

    # Рисуем граф
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000)
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}  # Метки стоимостей на ребрах
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Initial Transport Graph")
    plt.show()

    # Если решение предоставлено, рисуем граф с оптимальным решением
    if solution is not None:
        plt.figure(figsize=(8, 6))
        G_solution = nx.DiGraph()  # Создаем новый граф для решения

        # Добавляем узлы для источников и стоков
        for i in range(m):
            G_solution.add_node(f"S{i}")
        for j in range(n):
            G_solution.add_node(f"D{j}")

        # Добавляем только те ребра, которые используются в оптимальном решении
        for i in range(m):
            for j in range(n):
                if solution[i, j] > 0:  # Если перевозка больше нуля
                    G_solution.add_edge(f"S{i}", f"D{j}", weight=solution[i, j], label=f"{solution[i, j]:.0f}")

        # Рисуем граф с оптимальным решением
        nx.draw(G_solution, pos, with_labels=True, node_color='lightgreen', edge_color='blue', node_size=2000)
        edge_labels = {(u, v): d['label'] for u, v, d in G_solution.edges(data=True)}  # Метки перевозок на ребрах
        nx.draw_networkx_edge_labels(G_solution, pos, edge_labels=edge_labels)
        plt.title("Optimal Transport Solution")
        plt.show()


# Основная программа
m, n = 4, 5  # Количество источников и стоков
supply, demand, cost = generate_transport_problem(m, n)  # Генерируем транспортную задачу
solution = solve_transportation(supply, demand, cost)  # Решаем задачу
draw_graph(supply, demand, cost, solution)  # Визуализируем результат