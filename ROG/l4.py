import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import networkx as nx
import time


productTypes = 3  # Количество типов товаров
rawTypes = 2  # Количество типов сырья
days = 5  # Количество дней
N = 12  # Количество узлов в графе (1->(2,3)->(4,5)->6)

# Структура графа
source_node = 1
sink_node = N
arcs = [
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),

    (2, 7), (2, 8), (2, 9), (2, 10), (2, 11),
    (3, 7), (3, 8), (3, 9), (3, 10), (3, 11),
    (4, 7), (4, 8), (4, 9), (4, 10), (4, 11),
    (5, 7), (5, 8), (5, 9), (5, 10), (5, 11),
    (6, 7), (6, 8), (6, 9), (6, 10), (6, 11),

    (7, 12), (8, 12), (9, 12), (10, 12), (11, 12)
]

# --- Параметры модели ---
p_km = np.array([
    [10, 11, 10, 12, 11], # K=0
    [15, 14, 16, 15, 17], # K=1
    [ 9,  9, 10,  8, 10]  # K=2
]) # K x M

A_lk = np.array([
    [2, 3, 1], # L=0
    [1, 2, 4]  # L=1
]) # L x K

b_l0 = np.array([100, 120]) # L

gamma_lm = np.array([
    [10, 10, 15, 10, 5], # L=0
    [ 5,  8,  5,  8, 10] # L=1
]) # L x M

# d_ij – пропускная способность дуги
d_ij = np.zeros((N, N))
capacities = {
    (1, 2): 50, (1, 3): 60, (1, 4): 50, (1, 5): 60, (1, 6): 50,

    (2, 7): 40, (2, 8): 30, (2, 9): 35, (2, 10): 45, (2, 11): 30,
    (3, 7): 35, (3, 8): 45, (3, 9): 40, (3, 10): 30, (3, 11): 35,
    (4, 7): 40, (4, 8): 30, (4, 9): 35, (4, 10): 45, (4, 11): 30,
    (5, 7): 35, (5, 8): 45, (5, 9): 40, (5, 10): 30, (5, 11): 35,
    (6, 7): 40, (6, 8): 30, (6, 9): 35, (6, 10): 45, (6, 11): 30,

    (7, 12): 70, (8, 12): 65, (9, 12): 70, (10, 12): 65, (11, 12): 70
}
for u, v in arcs: d_ij[u-1, v-1] = capacities[(u, v)]

Q_k = np.array([50, 60, 40]) # K

# c_ij – стоимость перевозки (фиксированная)
c_ij = np.zeros((N, N))
costs = {
    (1, 2): 2, (1, 3): 3, (1, 4): 2, (1, 5): 3, (1, 6): 2,

    (2, 7): 4, (2, 8): 3, (2, 9): 3, (2, 10): 4, (2, 11): 3,
    (3, 7): 3, (3, 8): 4, (3, 9): 3, (3, 10): 3, (3, 11): 4,
    (4, 7): 4, (4, 8): 3, (4, 9): 3, (4, 10): 4, (4, 11): 3,
    (5, 7): 3, (5, 8): 4, (5, 9): 3, (5, 10): 3, (5, 11): 4,
    (6, 7): 4, (6, 8): 3, (6, 9): 3, (6, 10): 4, (6, 11): 3,

    (7, 12): 5, (8, 12): 4, (9, 12): 5, (10, 12): 4, (11, 12): 5
}
for u, v in arcs: c_ij[u-1, v-1] = costs[(u, v)]

pos = {
    1: np.array([0, 0]),
    2: np.array([1, 2]), 3: np.array([1, 1]), 4: np.array([1, 0]), 5: np.array([1, -1]), 6: np.array([1, -2]),
    7: np.array([2, 2]), 8: np.array([2, 1]), 9: np.array([2, 0]), 10: np.array([2, -1]), 11: np.array([2, -2]),
    12: np.array([3, 0])
}

# M_big – большая константа
max_possible_flow = np.sum(b_l0) + np.sum(gamma_lm)
M_big = max_possible_flow * 1.1

# --- Формулировка задачи для linprog ---

# Количество переменных
num_x = productTypes * days
num_lambda = N * N
num_z = N * N
total_vars = num_x + num_lambda + num_z

# Индексация
def get_x_idx(k, m): return k * days + m
def get_lambda_idx(i, j): return num_x + i * N + j
def get_z_idx(i, j): return num_x + num_lambda + i * N + j

# -------------- Целевая функция --------------
c_objective = np.zeros(total_vars)
for k in range(productTypes):
    for m in range(days):
        c_objective[get_x_idx(k, m)] = -p_km[k, m] # -p_km[k, m] для максимизации (особенность ЛП)

for i in range(N):
    for j in range(N):
        if (i + 1, j + 1) in arcs:
             c_objective[get_lambda_idx(i, j)] = c_ij[i, j]

# -------------- Матрицы ограничений --------------
A_eq_list = []
b_eq_list = []
A_ub_list = []
b_ub_list = []

# --- Ограничение на баланс производства и перевозки из начальной точки (3) ---
row = np.zeros(total_vars)
for k in range(productTypes):
    for m in range(days):
        row[get_x_idx(k, m)] = 1
for j in range(N):
    if (source_node, j + 1) in arcs:
        row[get_z_idx(source_node - 1, j)] = -1
A_eq_list.append(row)
b_eq_list.append(0)

# --- Баланс потока в промежуточных узлах (4) ---
intermediate_nodes = range(1, N - 1) # индексы 1 до N-2
for i in intermediate_nodes:
    row = np.zeros(total_vars)
    for prev_j in range(N):
         if (prev_j + 1, i + 1) in arcs: row[get_z_idx(prev_j, i)] = 1
    for next_j in range(N):
         if (i + 1, next_j + 1) in arcs: row[get_z_idx(i, next_j)] = -1
    A_eq_list.append(row)
    b_eq_list.append(0)

# --- Ограничение на использование сырья (2) ---
for l in range(rawTypes):
    cumulative_x_coeffs = np.zeros(num_x)
    cumulative_gamma = b_l0[l]
    for m in range(days):
        for k in range(productTypes):
            cumulative_x_coeffs[get_x_idx(k, m)] += A_lk[l, k]
        cumulative_gamma += gamma_lm[l, m]
        row = np.zeros(total_vars)
        row[:num_x] = cumulative_x_coeffs
        A_ub_list.append(row)
        b_ub_list.append(cumulative_gamma)

# --- Ограничение на спрос (5) ---
for k in range(productTypes):
    row = np.zeros(total_vars)
    for m in range(days): row[get_x_idx(k, m)] = 1
    A_ub_list.append(row)
    b_ub_list.append(Q_k[k])

# --- Ограничение на пропускную способность дуг (6) ---
for i in range(N):
    for j in range(N):
        if (i + 1, j + 1) in arcs:
            row = np.zeros(total_vars)
            row[get_z_idx(i, j)] = 1
            A_ub_list.append(row)
            b_ub_list.append(d_ij[i, j])

# --- Логическое ограничение на факт перевозки (7) ---
for i in range(N):
    for j in range(N):
         if (i + 1, j + 1) in arcs:
            row = np.zeros(total_vars)
            row[get_z_idx(i, j)] = 1
            row[get_lambda_idx(i, j)] = -M_big
            A_ub_list.append(row)
            b_ub_list.append(0)

# Преобразование в numpy массивы
A_eq = np.array(A_eq_list) if A_eq_list else None
b_eq = np.array(b_eq_list) if b_eq_list else None
A_ub = np.array(A_ub_list) if A_ub_list else None
b_ub = np.array(b_ub_list) if b_ub_list else None

# --- Границы переменных BOUNDS ---
bounds = []
for _ in range(num_x): bounds.append((0, None)) # x_km >= 0
for i in range(N):
    for j in range(N):
         bounds.append((0, 1) if (i+1, j+1) in arcs else (0, 0)) # lambda_ij: 0<=l<=1 for arcs, else 0
for i in range(N):
    for j in range(N):
         bounds.append((0, None) if (i+1, j+1) in arcs else (0, 0)) # z_ij >= 0 for arcs, else 0

# --- Указание целочисленности ---
integrality_vector = np.zeros(total_vars, dtype=int)

# x_km переменные - целочисленные
integrality_vector[0 : num_x] = 1

# lambda_ij переменные - бинарные (integrality=1 + bounds=(0,1))
for i in range(N):
    for j in range(N):
        if (i + 1, j + 1) in arcs:
            integrality_vector[get_lambda_idx(i, j)] = 1

# z_ij переменные - целочисленные
for i in range(N):
    for j in range(N):
        if (i + 1, j + 1) in arcs:
            integrality_vector[get_z_idx(i, j)] = 1

#4.3 Решение задачи с помощью linprog и параметра integrality

print("Запуск решения задачи MILP с помощью linprog(..., integrality=...)")
print("(Используется MILP-солвер HiGHS через SciPy)")
start_time = time.time()

# Передаем вектор integrality в linprog
result = linprog(c_objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, integrality=integrality_vector, method='highs')
end_time = time.time()
print(f"Решение заняло: {end_time - start_time:.4f} секунд")

# --- Анализ результата ---
print("\n--- Результат linprog (MILP) ---")
print(f"Статус: {result.message}") # Сообщение может отличаться для MILP

# Статусы для MILP в HiGHS могут быть другими, проверяем success
if result.success:
    optimal_value = -result.fun # Возвращаем к задаче максимизации
    print(f"Оптимальное значение целевой функции (максимизация прибыли): {optimal_value:.2f}")

    # Извлечение переменных (теперь они должны быть целочисленными/бинарными)
    x_km_opt = np.zeros((productTypes, days))
    lambda_ij_opt = np.zeros((N, N))
    z_ij_opt = np.zeros((N, N))

    sol_x = result.x
    for k in range(productTypes):
        for m in range(days):
            # Округляем до ближайшего целого из-за возможных малых погрешностей солвера
            x_km_opt[k, m] = np.round(sol_x[get_x_idx(k, m)])

    for i in range(N):
        for j in range(N):
             val = sol_x[get_lambda_idx(i, j)]
             lambda_ij_opt[i, j] = np.round(val) # Должны быть 0 или 1

    for i in range(N):
        for j in range(N):
             val = sol_x[get_z_idx(i, j)]
             z_ij_opt[i, j] = np.round(val) # Должны быть целыми

    # Расчет запасов b_lm ретроспективно
    b_lm_calc = np.zeros((rawTypes, days + 1))
    b_lm_calc[:, 0] = b_l0
    consumption_lm = np.zeros((rawTypes, days))
    for l in range(rawTypes):
        for m in range(days):
             # Используем округленные x_km_opt для расчета потребления
             consumption_lm[l, m] = np.sum(A_lk[l, :] * x_km_opt[:, m])

    for m in range(days):
        b_lm_calc[:, m+1] = b_lm_calc[:, m] - consumption_lm[:, m] + gamma_lm[:, m]
        b_lm_calc[:, m+1] = np.maximum(b_lm_calc[:, m+1], 0) # Гарантируем неотрицательность

    # --- Вывод некоторых результатов ---
    print("\nОптимальные объемы производства x_km (Целочисленные):")
    print(x_km_opt.astype(int))
    print("\nОптимальные факты перевозок lambda_ij (Бинарные):")
    print(lambda_ij_opt.astype(int))
    print("\nОптимальные объемы перевозок z_ij (Целочисленные):")
    print(z_ij_opt.astype(int))
    print("\nРасчетные запасы на КОНЕЦ дня b_lm (l-строка, m=1..M):")
    print(np.round(b_lm_calc[:, 1:]).astype(int)) # Запасы тоже должны быть целыми


    # Граф транспортной сети
    G = nx.DiGraph()
    node_labels = {i: str(i) for i in range(1, N + 1)}
    G.add_nodes_from(range(1, N + 1))
    pos = {
        1: np.array([0, 0]),  # Источник
        2: np.array([1, 2]), 3: np.array([1, 1]), 4: np.array([1, 0]), 5: np.array([1, -1]), 6: np.array([1, -2]),
        # Вторая группа
        7: np.array([2, 2]), 8: np.array([2, 1]), 9: np.array([2, 0]), 10: np.array([2, -1]), 11: np.array([2, -2]),
        # Третья группа
        12: np.array([3, 0])  # Сток
    }

    # Граф транспортный начальный
    plt.figure(figsize=(20, 10))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_list_init = []
    edge_labels_init = {}
    # Используем исходные 1-индексированные дуги для меток
    for u, v in arcs:
        edge_list_init.append((u, v))
        edge_labels_init[(u, v)] = f"d={d_ij[u-1, v-1]:.0f}  c={c_ij[u-1, v-1]:.0f}"
    nx.draw_networkx_edges(G, pos, edgelist=edge_list_init, edge_color='gray', arrows=True, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_init, font_size=8, alpha=0.7, label_pos=0.3, rotate=False)
    plt.title(r'Начальный транспортный граф ($\sum_j z_{ij} = \sum_j z_{ji}$)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # --- 4.4.2. Граф транспортное решение (MILP) ---
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=800)
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_list_sol = []
    edge_labels_sol = {}
    edge_widths_sol = []
    min_width = 1
    max_width = 8
    active_flows = [z_ij_opt[i, j] for i in range(N) for j in range(N) if (i+1,j+1) in arcs and z_ij_opt[i, j] > 0.1]
    max_flow_on_graph = max(active_flows) if active_flows else 1

    d_ij_dict = {(u-1, v-1): cap for (u,v), cap in capacities.items()}
    c_ij_dict = {(u-1, v-1): cost for (u,v), cost in costs.items()}

    for u_idx in range(N):
       for v_idx in range(N):
          if (u_idx + 1, v_idx + 1) in arcs:
             u, v = u_idx + 1, v_idx + 1
             flow = z_ij_opt[u_idx, v_idx]
             lmbda = lambda_ij_opt[u_idx, v_idx]

             if flow > 0.1:
                 edge_list_sol.append((u, v))
                 edge_labels_sol[(u, v)] = f"z={flow:.0f}\n(λ={lmbda:.0f})"
             elif lmbda > 0.1:
                 nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], edge_color='red', style='dashed', arrows=True)
                 edge_labels_sol[(u, v)] = f"z=0  (λ=1)"

    nx.draw_networkx_edges(G, pos, edgelist=edge_list_sol, node_size=800,
                            arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_sol, font_size=6, alpha=0.7, label_pos=0.2)
    plt.title(r'Граф решения (MILP)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    days = np.arange(1, days + 1)

    # 4.3 Запасы сырья b_lm (на конец дня m)
    plt.figure(figsize=(10, 6))
    for l in range(rawTypes):
        plt.plot(days, b_lm_calc[l, 1:], marker='o', linestyle='-', label=f'Сырье {l+1}')
    plt.xlabel("День (m)")
    plt.ylabel("Объем на складе (на конец дня)")
    plt.title(r'График сырья на складе ($b_{lm}$ - конец дня)')
    plt.xticks(days)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4.4 Суммарные запасы сырья (на конец дня m)
    plt.figure(figsize=(10, 6))
    total_b_lm = np.sum(b_lm_calc[:, 1:], axis=0)
    plt.plot(days, total_b_lm, marker='s', linestyle='-', color='yellow', label='Суммарный запас')
    plt.xlabel("День (m)")
    plt.ylabel("Суммарный объем на складе (на конец дня)")
    plt.title(r'График суммарных объемов сырья ($\sum_l b_{lm}$ - конец дня)')
    plt.xticks(days)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4.5 Объемы производства x_km
    plt.figure(figsize=(10, 6))
    for k in range(productTypes):
        plt.bar(days + k*0.2 - 0.2, x_km_opt[k, :].astype(int), width=0.2, label=f'Продукт K{k+1}') # Используем bar для целых
    plt.xlabel("День (m)")
    plt.ylabel("Объем производства")
    plt.title(r'График объемов производства каждый день ($x_{km}$)')
    plt.xticks(days)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4.6 Суммарные объемы производства sum_m x_km
    plt.figure(figsize=(10, 6))
    total_x_k = np.sum(x_km_opt, axis=1)
    product_types = [f'Продукт K{k+1}' for k in range(productTypes)]
    plt.bar(product_types, total_x_k.astype(int), color=['skyblue', 'orange', 'green'])
    plt.xlabel("Тип продукции (k)")
    plt.ylabel("Суммарный объем производства")
    plt.title(r'График объемов производства за все время ($\sum_m x_{km}$)')
    for index, value in enumerate(total_x_k):
        plt.text(index, value + 0.5, f'{value:.0f}', ha='center')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()