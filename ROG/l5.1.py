import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Параметры
N = 10 + 24  # Число вершин (n = 24)

# Генерация случайного сильно связного ориентированного графа
np.random.seed(42)
G = nx.DiGraph()
G.add_nodes_from(range(N))

# Создание базового неориентированного полного графа
G_undirected = nx.Graph()
G_undirected.add_nodes_from(range(N))

# Добавление ребер между каждой парой вершин
for i in range(N):
    for j in range(i + 1, N):  # Избегаем дублирования ребер и петель
        weight = np.random.randint(1, 10)  # Случайный вес ребра
        G_undirected.add_edge(i, j, weight=weight)

# Убедимся, что граф связный (он гарантированно связный, так как это полный граф)
assert nx.is_connected(G_undirected), "Граф должен быть связным"

# Преобразование неориентированного графа в ориентированный
for u, v, data in G_undirected.edges(data=True):
    if np.random.rand() < 0.5:  # Направление выбирается случайно
        G.add_edge(u, v, weight=data['weight'])
    else:
        G.add_edge(v, u, weight=data['weight'])

# Добавление случайных дуг для усиления связности
for i in range(N):
    for j in range(N):
        if i != j and not G.has_edge(i, j) and np.random.rand() < 0.05:  # Вероятность добавления дуги
            weight = np.random.randint(1, 10)  # Вес дуги
            G.add_edge(i, j, weight=weight)

# Убедимся, что граф сильно связный
if not nx.is_strongly_connected(G):
    print("Граф не является сильно связным. Добавление недостающих дуг...")
    # Добавляем недостающие дуги для обеспечения сильной связности
    components = list(nx.strongly_connected_components(G))
    while len(components) > 1:
        comp1, comp2 = components[0], components[1]
        u = next(iter(comp1))
        v = next(iter(comp2))
        weight = np.random.randint(1, 10)
        G.add_edge(u, v, weight=weight)
        components = list(nx.strongly_connected_components(G))

assert nx.is_strongly_connected(G), "Граф должен быть сильно связным"

# Получение списка дуг и их весов
edges = list(G.edges(data='weight'))
num_edges = len(edges)

# Построение минимального остовного дерева (МОД)
T_undirected = nx.minimum_spanning_tree(G.to_undirected(), weight='weight')

# Выбор корня графа (например, вершина 0)
root = 0

# Преобразование неориентированного МОД в ориентированный граф
T = nx.DiGraph()
T.add_nodes_from(T_undirected.nodes)

# Функция для рекурсивной ориентации ребер
def orient_edges(node, visited):
    visited.add(node)
    for neighbor in T_undirected.neighbors(node):
        if neighbor not in visited:
            T.add_edge(node, neighbor, weight=T_undirected[node][neighbor]['weight'])
            orient_edges(neighbor, visited)

# Начинаем ориентацию с корня
visited = set()
orient_edges(root, visited)

# Визуализация
pos = nx.spring_layout(G)  # Получаем позиции вершин для обоих графов

# Рисуем исходный граф серыми дугами
plt.figure(figsize=(16, 16))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color='lightgray', width=0.5)


plt.title("Исходный граф")

# Наложение МОД на исходный граф
node_colors = ['red' if node == root else 'darkgray' for node in T.nodes]
nx.draw_networkx_nodes(T, pos, node_color=node_colors, node_size=500)
nx.draw_networkx_edges(T, pos, edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=16, width=1)
nx.draw_networkx_labels(T, pos, font_size=10)

# Отображаем веса ребер МОД
edge_labels_mod = nx.get_edge_attributes(T, 'weight')
nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels_mod, font_color='green', font_size=8)

# Отображение результата
plt.title("Исходный граф и минимальное остовное дерево (МОД)")
plt.show()

# Вывод результатов
print("Корень графа:", root)
print("Вес минимального остовного дерева:", sum(edge[2]['weight'] for edge in T.edges(data=True)))
print("Дуги минимального остовного дерева:", list(T.edges(data=True)))