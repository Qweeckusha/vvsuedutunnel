from math import comb, floor, exp, factorial, sqrt, pi

from scipy.stats import norm

print('-------------------- 1 --------------------')
# Задача:
# В среднем 20% пакетов акций на аукционах продаются по первоначально
# заявленной цене. Найти вероятность того, что из 9 пакетов акций в результате торгов
# по первоначально заявленной цене: 1) не будут проданы 5 пакетов; 2) будет продано:
# а) менее 2 пакетов; б) не более 2; в) хотя бы 2 пакета; г) наивероятнейшее число
# пакетов

# Параметры задачи
n = 9  # количество испытаний
p = 0.2  # вероятность успеха
q = 1 - p  # вероятность неудачи

# Функция для вычисления вероятности P(X = k)
def binomial_probability(n, k, p):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# 1) Вероятность того, что не будут проданы 5 пакетов (X = 4)
prob_1 = binomial_probability(n, 4, p)

# 2a) Вероятность того, что будет продано менее 2 пакетов (X < 2)
prob_2a = sum(binomial_probability(n, k, p) for k in range(0, 2))

# 2b) Вероятность того, что будет продано не более 2 пакетов (X <= 2)
prob_2b = sum(binomial_probability(n, k, p) for k in range(0, 3))

# 2c) Вероятность того, что будет продано хотя бы 2 пакета (X >= 2)
prob_2c = 1 - prob_2a  # дополнение к вероятности "менее 2"

# 2d) Наивероятнейшее число проданных пакетов
# Наивероятнейшее число k определяется как целое число, ближайшее к (9 + 1) * 0.2
most_probable_k = round((n + 1) * p)

# Вывод результатов
print(f"1) Вероятность того, что не будут проданы 5 пакетов: {prob_1:.4f}")
print(f"2a) Вероятность того, что будет продано менее 2 пакетов: {prob_2a:.4f}")
print(f"2b) Вероятность того, что будет продано не более 2 пакетов: {prob_2b:.4f}")
print(f"2c) Вероятность того, что будет продано хотя бы 2 пакета: {prob_2c:.4f}")
print(f"2d) Наивероятнейшее число проданных пакетов: {most_probable_k}")

print('\n-------------------- 2 --------------------')
# Задача: Сколько раз необходимо подбросить игральную кость,
# чтобы наивероятнейшее выпадение тройки было равно 10?

# Вероятность выпадения тройки при одном подбрасывании
p = 1 / 6

# Наивероятнейшее число успехов
k_0 = 10

# Формула для нахождения минимального n:
# (n + 1) * p - 1 <= k_0 <= (n + 1) * p
# Решаем неравенство относительно n
n_min = int((k_0 + 1) / p - 1)  # Минимальное значение n

# Проверка результата
# При n = 65:
n = n_min
lower_bound = (n + 1) * p - 1  # Левая граница неравенства
upper_bound = (n + 1) * p      # Правая граница неравенства

# Вывод результатов с формулами и числовыми данными в комментариях
print(f"Минимальное количество подбрасываний n: {n}")
# Для n = 65:
# Левая граница: (n + 1) * p - 1 = (65 + 1) * (1/6) - 1 = 10
# Правая граница: (n + 1) * p = (65 + 1) * (1/6) = 11
print(f"Проверка: Левая граница = {lower_bound:.1f}, Правая граница = {upper_bound:.1f}")

print('\n-------------------- 3 --------------------')

# Задача: Сколько следует сыграть партий в шахматы с вероятностью победы в одной
# партии, равной 1/3, чтобы наивероятнейшее число побед было равно 5?

p = 1 / 3 # Вероятность победы в одной партии

k_0 = 5 # Наивероятнейшее число побед

# Формула для нахождения минимального n:
# (n + 1) * p - 1 <= k_0 <= (n + 1) * p
# Решаем неравенство относительно n
n_min = int((k_0 + 1) / p - 1)  # Минимальное значение n

# Проверка результата
# При n = 17:
n = n_min
lower_bound = (n + 1) * p - 1  # Левая граница неравенства
upper_bound = (n + 1) * p      # Правая граница неравенства

# Вывод результатов с формулами и числовыми данными в комментариях
print(f"Минимальное количество партий n: {n}")
# Для n = 17:
# Левая граница: (n + 1) * p - 1 = (17 + 1) * (1/3) - 1 = 5
# Правая граница: (n + 1) * p = (17 + 1) * (1/3) = 6
print(f"Проверка: Левая граница = {lower_bound:.1f}, Правая граница = {upper_bound:.1f}")

print('\n-------------------- 4 --------------------')
# Задача:
# Пусть вероятность того, что телевизор потребует ремонта в течение гарантийного
# срока, равна 0,2. Найти вероятность того, что в течение гарантийного срока из 6
# телевизоров: а) не более одного потребует ремонта; б) хотя бы один не потребует
# ремонта.
from math import comb

# Параметры задачи
n = 6  # количество телевизоров
p = 0.2  # вероятность того, что телевизор потребует ремонта
q = 1 - p  # вероятность того, что телевизор не потребует ремонта

# Функция для вычисления вероятности P(X = k)
def binomial_probability(n, k, p):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# а) Вероятность того, что не более одного телевизора потребует ремонта (X <= 1)
prob_a = sum(binomial_probability(n, k, p) for k in range(0, 2))  # P(X = 0) + P(X = 1)

# б) Вероятность того, что хотя бы один телевизор не потребует ремонта
# Это дополнение к вероятности того, что все телевизоры потребуют ремонта (X = 6)
prob_b = 1 - binomial_probability(n, 6, p)  # 1 - P(X = 6)

# Вывод результатов с формулами и числовыми данными в комментариях
print(f"а) Вероятность того, что не более одного телевизора потребует ремонта: {prob_a:.4f}")
# Для P(X = 0): C(6, 0) * (0.2)^0 * (0.8)^6 = 1 * 1 * 0.2621 = 0.2621
# Для P(X = 1): C(6, 1) * (0.2)^1 * (0.8)^5 = 6 * 0.2 * 0.3277 = 0.3932
# Итого: P(X <= 1) = 0.2621 + 0.3932 = 0.6553

print(f"б) Вероятность того, что хотя бы один телевизор не потребует ремонта: {prob_b:.4f}")
# Для P(X = 6): C(6, 6) * (0.2)^6 * (0.8)^0 = 1 * 0.000064 * 1 = 0.000064
# Итого: P(хотя бы один не потребует) = 1 - 0.000064 = 0.9999

print('\n-------------------- 5 --------------------')
# Задача:
# Что более вероятно выиграть у равносильного противника: не менее двух партий
# из трёх или не более одной из двух?

# Вероятность выиграть у равносильного противника = 0.5 (p = 0.5)
# Вероятность проиграть = 1 - p = 0.5 (q = 0.5)

# Формула для вероятности биномиального распределения:
# P(X = k) = C(n, k) * (p^k) * (q^(n-k))
# где C(n, k) - число сочетаний из n по k

from math import comb

# Переменные:
p = 0.5  # вероятность выигрыша одной партии
q = 1 - p  # вероятность проигрыша одной партии

# Случай 1: Не менее двух партий из трёх (X >= 2 при n = 3)
# P(X >= 2) = P(X = 2) + P(X = 3)
n1 = 3
prob_at_least_two_wins = (
    comb(n1, 2) * (p**2) * (q**(n1-2)) +  # P(X = 2)
    comb(n1, 3) * (p**3) * (q**(n1-3))    # P(X = 3)
)

# Случай 2: Не более одной партии из двух (X <= 1 при n = 2)
# P(X <= 1) = P(X = 0) + P(X = 1)
n2 = 2
prob_at_most_one_win = (
    comb(n2, 0) * (p**0) * (q**(n2-0)) +  # P(X = 0)
    comb(n2, 1) * (p**1) * (q**(n2-1))    # P(X = 1)
)

# Вывод результатов
print("Вероятность выиграть не менее двух партий из трёх:", prob_at_least_two_wins)
print("Вероятность выиграть не более одной партии из двух:", prob_at_most_one_win)

# Определение, что более вероятно
if prob_at_least_two_wins > prob_at_most_one_win:
    print("Более вероятно выиграть не менее двух партий из трёх.")
else:
    print("Более вероятно выиграть не более одной партии из двух.")

print('\n-------------------- 6 --------------------')
# Задача:
# Вероятность попадания в кольцо при штрафном броске для баскетболиста равна
# 0,8. Сколько надо произвести бросков, чтобы наивероятнейшее число попаданий
# было равно 20?

# Вероятность попадания
p = 0.8

# Наивероятнейшее число попаданий
k_0 = 20

# Формула для наивероятнейшего числа успехов:
# k_0 = floor((n + 1) * 0.8)
# Решаем неравенство: k_0 <= (n + 1) * 0.8 < k_0 + 1

# Нижняя граница:
# 20 <= (n + 1) * 0.8 -> n >= (20 / 0.8) - 1
n_min = int((k_0 / p) - 1)  # n_min = int((20 / 0.8) - 1) = int(25 - 1) = 24

# Верхняя граница:
# (n + 1) * 0.8 < 20 + 1 -> n < ((20 + 1) / 0.8) - 1
n_max = int(((k_0 + 1) / p) - 1)  # n_max = int(((20 + 1) / 0.8) - 1) = int(26.25 - 1) = 25

print(f"Количество бросков n должно быть в диапазоне: [{n_min}, {n_max}]")

# Проверяем, что при n = 25 наивероятнейшее число попаданий равно 20
n = 25
k_0_calculated = int((n + 1) * p)  # k_0_calculated = int((25 + 1) * 0.8) = int(26 * 0.8) = int(20.8) = 20
print(f"При n = {n} наивероятнейшее число попаданий: {k_0_calculated}")

print('\n-------------------- 7 --------------------')
# Задача:
# Пусть вероятность того, что студент опоздает на лекцию, равна 0,08.
# Найти наиболее вероятное число опоздавших из 96 студентов.

# Вероятность опоздания одного студента
p = 0.08

# Количество студентов
n = 96

# Наиболее вероятное число опоздавших (формула: k_0 = floor((n + 1) * p))
k_0 = floor((n + 1) * p)  # k_0 = floor((96 + 1) * 0.08) = floor(97 * 0.08) = floor(7.76) = 7 - округляем до целых

print(f"Наиболее вероятное число опоздавших: {k_0}")\

print('\n-------------------- 8 --------------------')
# Задача:
# В результате каждого визита страхового агента договор заключается с
# вероятностью 0,1. Найти наивероятнейшее число заключенных договоров после 25
# визитов.

# Вероятность заключения договора при одном визите
p = 0.1

# Количество визитов
n = 25

# Наивероятнейшее число заключенных договоров вычисляется по формуле:
# k_0 = floor((n + 1) * p)
# Подставляем значения:
# k_0 = floor((25 + 1) * 0.1)
# k_0 = floor(26 * 0.1)
# k_0 = floor(2.6)
# floor(2.6) = 2 (округление вниз до ближайшего целого числа)

k_0 = floor((n + 1) * p)  # Вычисляем значение

# Вывод результата
print(f"Наивероятнейшее число заключенных договоров: {k_0}")

print('\n-------------------- 9 --------------------')
# Задача:
# Найти вероятность того, что в 243 испытаниях событие А наступит ровно 70 раз,
# если вероятность наступления этого события в каждом испытании равна 0,25.

# Параметры задачи
n = 243  # количество испытаний
k = 70   # количество успехов
p = 0.25  # вероятность успеха в одном испытании

# Формула биномиального распределения:
# P(X = k) = C(n, k) * (p^k) * ((1-p)^(n-k))
# Подставляем значения:
# P(X = 70) = C(243, 70) * (0.25^70) * (0.75^(243-70))

probability = (
    comb(n, k) *                # C(243, 70)
    (p**k) *                    # (0.25^70)
    ((1 - p)**(n - k))          # (0.75^(243-70))
)

# Вывод результата
print(f"Вероятность того, что событие A наступит ровно {k} раз: {probability}")

print('\n-------------------- 10 --------------------')
# Задача:
# Фабрика выпускает 70% продукции первого сорта. Чему равна вероятность, что в
# партии из 1000 изделий число изделий первого сорта от 652 до 760

# Параметры задачи
n = 1000  # количество изделий
p = 0.7   # вероятность изделия быть первого сорта

# Математическое ожидание и стандартное отклонение
mu = n * p  # математическое ожидание
sigma = (n * p * (1 - p)) ** 0.5  # стандартное отклонение

# Интервал для числа изделий первого сорта
a = 652  # нижняя граница
b = 760  # верхняя граница

# Применяем поправку на непрерывность
a_corrected = a - 0.5  # нижняя граница с поправкой
b_corrected = b + 0.5  # верхняя граница с поправкой

# Стандартизация границ (перевод в Z-оценки)
z1 = (a_corrected - mu) / sigma  # Z-оценка для нижней границы
z2 = (b_corrected - mu) / sigma  # Z-оценка для верхней границы

# Вычисление вероятности через стандартное нормальное распределение
probability = norm.cdf(z2) - norm.cdf(z1)

# Вывод результата
print(f"Вероятность того, что число изделий первого сорта от {a} до {b}: {probability:.4f}")

print('\n-------------------- 11 --------------------')
# Задача:
# Известно, что процент брака для некоторой детали равен 0,5%. Контролер
# проверяет 1000 деталей. Какова вероятность обнаружить ровно три бракованные
# детали? Какова вероятность обнаружить не меньше трех бракованных деталей?

# Параметры задачи
n = 1000  # количество деталей
p = 0.005  # вероятность брака для одной детали

# λ для распределения Пуассона
λ = n * p  # λ = 1000 * 0.005 = 5

# Формула Пуассона:
# P(X = k) = (λ^k * e^(-λ)) / k!
def poisson_probability(k, λ):
    return (λ**k) * exp(-λ) / factorial(k)

# Задача 1: Вероятность обнаружить ровно три бракованные детали
k1 = 3
probability_exact_3 = poisson_probability(k1, λ)

# Задача 2: Вероятность обнаружить не меньше трех бракованных деталей
# P(X >= 3) = 1 - P(X < 3) = 1 - (P(X = 0) + P(X = 1) + P(X = 2))
probability_at_least_3 = 1
for k in range(3):  # Считаем для k = 0, 1, 2
    probability_at_least_3 -= poisson_probability(k, λ)

# Вывод результатов
print(f"Вероятность обнаружить ровно три бракованные детали: {probability_exact_3:.6f}")
print(f"Вероятность обнаружить не меньше трех бракованных деталей: {probability_at_least_3:.6f}")

print('\n-------------------- 12 --------------------')
# Задача:
# На факультете насчитывается 1825 студентов. Какова вероятность того, что 1
# сентября является днем рождения одновременно четырех студентов факультета?
from math import comb

# Параметры задачи
n = 1825  # количество студентов
p = 1 / 365  # вероятность того, что у студента день рождения 1 сентября
q = 1 - p  # вероятность того, что у студента день рождения не 1 сентября

# Формула биномиального распределения:
# P(X = k) = C(n, k) * (p^k) * (q^(n-k))
# где C(n, k) = n! / (k! * (n-k)!)

# Вычисляем вероятность того, что ровно у четырех студентов день рождения 1 сентября
k = 4
probability_exact_4 = comb(n, k) * (p**k) * (q**(n - k))

# Вывод результата
print(f"Вероятность того, что ровно у четырех студентов день рождения 1 сентября: {probability_exact_4:.6f}")

print('\n-------------------- 13 --------------------')
# Задача:
# Завод отправил на базу 10 000 стандартных изделий. Среднее число изделий, повреждаемых при транспортировке,
# составляет 0,02%. Найти вероятность того, что из 10 000 изделий: 1) будет повреждено: а) три изделия;
# б) по крайней мере три изделия; 2) не будет повреждено: а) 9997; б) хотя бы 9997

from math import exp, factorial

# Параметры задачи
n = 10000  # количество изделий
p = 0.0002  # вероятность повреждения одного изделия
λ = n * p  # параметр распределения Пуассона (λ = n * p = 10000 * 0.0002 = 2)

# Формула Пуассона:
# P(X = k) = (λ^k * e^(-λ)) / k!
def poisson_probability(k, λ):
    return (λ**k) * exp(-λ) / factorial(k)

# Задача 1: Вероятность того, что будет повреждено ровно три изделия
k1 = 3
probability_exact_3 = poisson_probability(k1, λ)  # P(X = 3)

# Задача 2: Вероятность того, что будет повреждено по крайней мере три изделия
# P(X >= 3) = 1 - P(X < 3) = 1 - (P(X = 0) + P(X = 1) + P(X = 2))
probability_at_least_3 = 1
for k in range(3):  # Считаем для k = 0, 1, 2
    probability_at_least_3 -= poisson_probability(k, λ)

# Задача 3: Вероятность того, что не будет повреждено ровно 9997 изделий
# Если повреждено ровно 3 изделия, то не повреждено ровно 9997
probability_not_damaged_9997 = probability_exact_3  # P(X = 3)

# Задача 4: Вероятность того, что не будет повреждено хотя бы 9997 изделий
# Если повреждено не более двух изделий, то не повреждено хотя бы 9997
# P(X <= 2) = P(X = 0) + P(X = 1) + P(X = 2)
probability_not_damaged_at_least_9997 = 0
for k in range(3):  # Считаем для k = 0, 1, 2
    probability_not_damaged_at_least_9997 += poisson_probability(k, λ)

# Вывод результатов
print(f"1а) Вероятность того, что будет повреждено ровно три изделия: {probability_exact_3:.6f}")
print(f"1б) Вероятность того, что будет повреждено по крайней мере три изделия: {probability_at_least_3:.6f}")
print(f"2а) Вероятность того, что не будет повреждено ровно 9997 изделий: {probability_not_damaged_9997:.6f}")
print(f"2б) Вероятность того, что не будет повреждено хотя бы 9997 изделий: {probability_not_damaged_at_least_9997:.6f}")

print('\n-------------------- 14 --------------------')
# Задача:
# По результатам проверок налоговыми инспекциями установлено, что в среднем
# каждое второе малое предприятие региона имеет нарушение финансовой
# дисциплины. Найти вероятность того, что из 1000 зарегистрированных в регионе
# малых предприятий имеют нарушения финансовой дисциплины а) 480 предприятий,
# б) наивероятнейшее число предприятий; в) не менее 480; г) от 480 до 520


# Параметры задачи
n = 1000  # количество предприятий
p = 0.5   # вероятность нарушения финансовой дисциплины
q = 1 - p # вероятность отсутствия нарушения

# Параметры нормального распределения
mu = n * p  # математическое ожидание
sigma = sqrt(n * p * q)  # стандартное отклонение

# Формула Лапласа для P(X = k)
def laplace_local(k, mu, sigma):
    return (1 / (sqrt(2 * pi) * sigma)) * exp(-((k - mu)**2) / (2 * sigma**2))

# Задача а) Вероятность того, что ровно 480 предприятий имеют нарушения
k_a = 480
probability_exact_480 = laplace_local(k_a, mu, sigma)

# Задача б) Наивероятнейшее число предприятий
# Наивероятнейшее число k_0 вычисляется как floor((n + 1) * p)
k_0 = int((n + 1) * p)  # k_0 = floor(1001 * 0.5) = 500

# Задача в) Вероятность того, что не менее 480 предприятий имеют нарушения
# Для этого случая используем интегральную функцию Лапласа
from scipy.stats import norm

# Интегральная функция Лапласа с поправкой на непрерывность
def normal_approximation(a, b, mu, sigma):
    z1 = (a - 0.5 - mu) / sigma  # нижняя граница с поправкой на непрерывность
    z2 = (b + 0.5 - mu) / sigma  # верхняя граница с поправкой на непрерывность
    return norm.cdf(z2) - norm.cdf(z1)

# Вероятность не менее 480 нарушений
probability_at_least_480 = 1 - normal_approximation(0, 479, mu, sigma)

# Задача г) Вероятность того, что от 480 до 520 предприятий имеют нарушения
# P(480 <= X <= 520)
probability_between_480_and_520 = normal_approximation(480, 520, mu, sigma)

# Вывод результатов
print(f"а) Вероятность того, что ровно 480 предприятий имеют нарушения: {probability_exact_480:.6f}")
print(f"б) Наивероятнейшее число предприятий с нарушениями: {k_0}")
print(f"в) Вероятность того, что не менее 480 предприятий имеют нарушения: {probability_at_least_480:.6f}")
print(f"г) Вероятность того, что от 480 до 520 предприятий имеют нарушения: {probability_between_480_and_520:.6f}")

print('\n-------------------- 15 --------------------')
# Задача:
# В страховой компании 10 тыс. клиентов. Страховой взнос каждого клиента
# составляет 500 руб. При наступлении страхового случая, вероятность которого по
# имеющимся данным и оценкам экспертов можно считать равной p = 0,005, страховая
# компания обязана выплатить клиенту страховую сумму размером 50 тыс. руб. На
# какую прибыль может рассчитывать страховая компания с надежностью 0,95?

from math import sqrt
from scipy.stats import norm

# Параметры задачи
n = 10000  # количество клиентов
p = 0.005  # вероятность наступления страхового случая
q = 1 - p  # вероятность отсутствия страхового случая (q = 1 - 0.005 = 0.995)

# Страховой взнос и выплата
insurance_payment = 500  # страховой взнос (в рублей)
compensation = 50000  # страховая выплата при наступлении случая (в рублей)

# Математическое ожидание и стандартное отклонение
# μ = n * p = 10000 * 0.005 = 50
mu = n * p

# Стандартное отклонение:
# σ = sqrt(n * p * q) = sqrt(10000 * 0.005 * 0.995) ≈ sqrt(49.75) ≈ 7.05
sigma = sqrt(n * p * q)

# Надежность 0.95 соответствует z = 1.645 (критическое значение из таблицы нормального распределения)
z = norm.ppf(0.95)

# Находим максимальное число случаев k_max с вероятностью 0.95
# k_max = μ + z * σ = 50 + 1.645 * 7.05 ≈ 50 + 11.61 ≈ 61
k_max = int(mu + z * sigma)

# Вычисляем общую сумму поступлений (всего поступило от клиентов)
# total_income = n * insurance_payment = 10000 * 500 = 5000000
total_income = n * insurance_payment

# Вычисляем максимальные выплаты
# max_payouts = k_max * compensation = 61 * 50000 = 3050000
max_payouts = k_max * compensation

# Рассчитываем прибыль
# profit = total_income - max_payouts = 5000000 - 3050000 = 1950000
profit = total_income - max_payouts

# Вывод результатов
print(f"Максимальное число страховых случаев с вероятностью 0.95: {k_max}")
print(f"Общая сумма поступлений: {total_income} руб.")
print(f"Максимальные выплаты: {max_payouts} руб.")
print(f"Прибыль страховой компании с надежностью 0.95: {profit} руб.")
