import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Интегральная F(x)
def F(x):
    if x <= 0:
        return 0
    if 0 < x <= 4:
        return (x**2) / 16
    if x > 4:
        return 1

# Дифференциальная f(x)
def f(x):
    # F(X) = 0 => F'(X) = f(x) = 0
    # F(X) = 1 => F'(X) = f(x) = 0
    if x <= 0 or x > 4:
        return 0
    # F(X) = x**2/16 => F'(X) = f(x) = 2x/16 = x/8
    else:
        return x / 8

# M(X) = ∫^∞_−∞ xf(x)dx
def math_wait():
    integral = lambda  x: x * f(x)
    return quad(integral, 0, 4)[0]

#
def disperse(mean_value):
    integral = lambda x: x**2 * f(x)
    mean_square = quad(integral, 0, 4)[0]
    return mean_square - mean_value**2

# График F(X)
x_values = np.linspace(-2, 6, 500)
F_values = [F(x) for x in x_values]

plt.figure(figsize=(10,6))
plt.plot(x_values, F_values, label="F(x)", color="green")
plt.title("1) F(X)")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.grid()
plt.show()

# График f(x)
f_values = [f(x) for x in x_values]

plt.figure(figsize=(10,6))
plt.plot(x_values, f_values, label="f(x)", color="green")
plt.title("1) f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

print("Task 1")
print(f"Математическое ожидание M(X): {math_wait():.4f}")
print(f"Дисперсия D(X): {disperse(math_wait()):.4f}")


# Интегральная F(x)
def F(x):
    if x <= 0:
        return 0
    if 0 < x <= 9:
        return (x**2) / 81
    if x > 9:
        return 1

# Дифференциальная f(x)
def f(x):
    # F(X) = 0 => F'(X) = f(x) = 0
    # F(X) = 1 => F'(X) = f(x) = 0
    if x <= 0 or x > 9:
        return 0
    # F(X) = x**2/81 => F'(X) = f(x) = 2x/81
    else:
        return (2*x) / 81

# M(X) = ∫^∞_−∞ xf(x)dx
def math_wait():
    integral = lambda x: x * f(x)
    return quad(integral, 0, 9)[0]

#
def disperse(mean_value):
    integral = lambda x: x**2 * f(x)
    mean_square = quad(integral, 0, 9)[0]
    return mean_square - mean_value**2

# График F(X)
x_values = np.linspace(-3, 12, 500)
F_values = [F(x) for x in x_values]

plt.figure(figsize=(10,6))
plt.plot(x_values, F_values, label="F(x)", color="green")
plt.title("2) F(X)")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.grid()
plt.show()

# График f(x)
f_values = [f(x) for x in x_values]

plt.figure(figsize=(10,6))
plt.plot(x_values, f_values, label="f(x)", color="green")
plt.title("2) f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

print("Task 2")
print(f"Математическое ожидание M(X): {math_wait():.4f}")
print(f"Дисперсия D(X): {disperse(math_wait()):.4f}")