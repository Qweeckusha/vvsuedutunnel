import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Параметры
N = 10 + 24  # Число вершин

# Генерация полного графа
np.random.seed(42)
G = nx.Graph()
G.add_nodes_from(range(N))

# Добавляем ребра между каждой парой вершин
for i in range(N):
    for j in range(i + 1, N):  # Избегаем дублирования ребер и петель
        weight = np.random.randint(2, 20)  # Случайный вес ребра
        G.add_edge(i, j, weight=weight)

assert nx.is_connected(G), "Граф должен быть связным"

# Построение минимального остовного дерева (МОД) с использованием алгоритма Краскала
T = nx.minimum_spanning_tree(G, weight='weight')

# ==== Визуализация ====
pos = nx.spring_layout(G)  # Получаем позиции вершин для обоих графов

plt.figure(figsize=(20, 16))
nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=200, edge_color='lightgray', width=1)

# Отображаем веса ребер исходного графа
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=16)

# Наложение МОД на исходный граф
nx.draw_networkx_nodes(T, pos, node_color='black', node_size=600)
nx.draw_networkx_edges(T, pos, edge_color='black', width=2)
nx.draw_networkx_labels(T, pos, font_size=14)

# Отображаем веса ребер МОД
edge_labels_mod = nx.get_edge_attributes(T, 'weight')
nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels_mod, font_color='black', font_size=16)

# Отображение результата
plt.title("Исходный граф и минимальное остовное дерево (МОД)")
plt.show()

# Вывод ребер и весов МОД
print("\nРебра минимального остовного дерева (неориентированный граф):")
total_weight = 0
for u, v, data in T.edges(data=True):
    weight = data['weight']
    print(f"{u + 1} - {v + 1}, вес: {weight}")
    total_weight += weight

print(f"Общая стоимость: {total_weight}")