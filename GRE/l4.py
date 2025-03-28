from math import comb

print('\n-------------------- 1 --------------------')
# Задача:
# В лотерее «Спортлото 6 из 45» денежные призы получают участники, угадавшие 3, 4, 5 и 6
# видов спорта из отобранных случайно 6 видов из 45 (размер приза увеличивается с увеличением
# числа угаданных видов спорта). Найти закон распределения случайной величины X — числа
# угаданных видов спорта среди случайно отобранных шести.

# Параметры задачи
N = 45  # общее количество видов спорта
M = 6   # количество выигрышных видов спорта
n = 6   # количество выбранных видов спорта

# Функция для вычисления вероятности по формуле гипергеометрического распределения
def hypergeometric_probability(N, M, n, k):
    if k > min(M, n) or k < max(0, n - (N - M)):
        return 0  # Вероятность равна нулю, если k вне допустимого диапазона
    return comb(M, k) * comb(N - M, n - k) / comb(N, n)

# Закон распределения случайной величины X
distribution = {}
for k in range(7):  # k может принимать значения от 0 до 6
    probability = hypergeometric_probability(N, M, n, k)
    distribution[k] = probability

# Вывод результата
print("Закон распределения случайной величины X:")
for k, prob in distribution.items():
    print(f"P(X = {k}) = {prob:.6f}")

