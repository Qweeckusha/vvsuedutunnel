# ИДЗ 2.1
# БИН-23-1
# Котов Александра Юрьевич
# Вариант 3

print("Task 1a")
X_list = [1, 2, 3]
p_X = [0.2, 0.1, 0.7]

Y_list = []
for i in range(0, len(X_list)):
    Y_list.append(X_list[i]**4)
# print(Y_list)

Math_wait = 0
# M(Y) = ∑y_i * P(Y=y_i)
for i in range(0, len(X_list)):
    Y = Y_list[i] * p_X[i]
    Math_wait += Y
print(f"Математическое ожидание M(Y): {round(Math_wait, ndigits=3)}")

Disperse = 0
# D(Y)=M(Y^2)−[M(Y)]^2
Math_wait_squared = 0
# M(Y^2) = ∑(y_i)^2 * P(Y=y_i)
for i in range(0, len(X_list)):
    Y = Y_list[i]**2 * p_X[i]
    Math_wait_squared += Y
# print(f"Математическое ожидание в квадрате M(Y^2): {round(Math_wait_squared, ndigits=3)}")
Disperse = Math_wait_squared - Math_wait**2
print(f"Дисперсия D(Y): {round(Disperse, ndigits=2)}\n")

print("Task 1b")
X_list = [-1, 0, 1]
p_X = [0.1, 0.2, 0.7]

Y_list = []
for i in range(0, len(X_list)):
    Y_list.append(X_list[i]**4)
# print(Y_list)

Math_wait = 0
# M(Y) = ∑y_i * P(Y=y_i)
for i in range(0, len(X_list)):
    Y = Y_list[i] * p_X[i]
    Math_wait += Y
print(f"Математическое ожидание M(Y): {round(Math_wait, ndigits=3)}")

Disperse = 0
# D(Y)=M(Y^2)−[M(Y)]^2
Math_wait_squared = 0
# M(Y^2) = ∑(y_i)^2 * P(Y=y_i)
for i in range(0, len(X_list)):
    Y = Y_list[i]**2 * p_X[i]
    Math_wait_squared += Y
# print(f"Математическое ожидание в квадрате M(Y^2): {round(Math_wait_squared, ndigits=3)}")
Disperse = Math_wait_squared - Math_wait**2
print(f"Дисперсия D(Y): {round(Disperse, ndigits=2)}\n")

print("Task 2")
X_list = [1, 2]
p_X = [0.2, 0.8]
Y_list = [0.5, 1]
p_Y = [0.3, 0.7]

Math_wait_X = 0
# M(X) = ∑y_i * P(Y=y_i)
for i in range(0, len(X_list)):
    X = X_list[i] * p_X[i]
    Math_wait_X += X
print(f"Математическое ожидание M(X): {round(Math_wait_X, ndigits=3)}")

Math_wait_Y = 0
# M(X) = ∑y_i * P(Y=y_i)
for i in range(0, len(Y_list)):
    Y = Y_list[i] * p_Y[i]
    Math_wait_Y += Y
print(f"Математическое ожидание M(Y): {round(Math_wait_Y, ndigits=3)}")

Math_wait_XY = 0
# Так как X и Y независимы, их совместное распределение можно найти как произведение маргинальных вероятностей
# M(X = x_i, Y = y_j) = P(X = x_i) * P(Y = y_j)

XY_list = [[],
           []]
for i in range(0, len(X_list)):
    for j in range(0, len(Y_list)):
        XY_list[i].append(round(p_X[i]*p_Y[j], ndigits=3))

print(XY_list)
for i in range(0, len(X_list)):
    for j in range(0, len(Y_list)):
        print(f"(X = {X_list[i]}, Y = {Y_list[j]}): {X_list[i]} * {Y_list[j]} * {XY_list[i][j]} = {X_list[i]*Y_list[j]*XY_list[i][j]}")
        Math_wait_XY += X_list[i]*Y_list[j]*XY_list[i][j]

print(f"Математическое ожидание M(XY): {round(Math_wait_XY, ndigits=3)}")
print(f"\nПроверяем")
print(f"M(X)*M(Y) = {Math_wait_X} * {Math_wait_Y} = {round(Math_wait_XY, ndigits=3)}")
print(f"M(XY) = {round(Math_wait_XY, ndigits=3)}")
print(f"M(X)*M(Y) = M(XY)\nПроверено")