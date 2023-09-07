import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def function(x, a, b):
    x = np.atleast_1d(x)  # Ensure x is an array, even if it's a single scalar
    result = []
    for xi in x:
        result.append(
            np.sum(
                [
                    np.maximum(0, -(4 / (b[i] - a[i]) ** 2) * (xi - a[i]) * (xi - b[i]))
                    for i in range(len(a))
                ]
            )
        )
    return np.array(result)


def p_k(X, mu_k, sigma_k):
    return (1 / (sigma_k * np.sqrt(2 * np.pi))) * np.exp(
        -((X - mu_k) ** 2) / (2 * sigma_k**2)
    )


def composite_trapezoidal(f, a, b, h):
    x = np.arange(a, b + h, h)
    y = f(x)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral


def integrated_function(x):
    x = np.asarray(x).reshape(-1, 1)
    return np.sum(np.maximum(0, -(4 / (b - a) ** 2) * (x - a) * (x - b)), axis=1)


# Initial parameters
N = 50
alfai = np.array([0.3333, 0.3333, 0.3333])
ni = np.round(alfai * N).astype(int)
N = ni[0] * len(alfai)
mu = np.array([2, 5, 7])
sigma = np.array([0.8, 0.8, 0.4]) / 2
a = mu - 2 * sigma
b = mu + 2 * sigma

# Compute MIS estimate
np.random.seed(8)
F = 0
sampled_points_X = []
sampled_points_Y = []
for i in range(len(alfai)):
    for j in range(ni[i]):
        X = sigma[i] * np.random.randn() + mu[i]
        Y = function(X, a, b)
        p_techo = sum(
            [(ni[k] / N) * p_k(X, mu[k], sigma[k]) for k in range(len(alfai))]
        )
        F = F + (Y / p_techo)
        sampled_points_X.append(X)
        sampled_points_Y.append(Y)
Imis = F / N

# Numerical integration
numerical_integral = sum(
    [
        quad(lambda x: max(0, -(4 / (bi - ai) ** 2) * (x - ai) * (x - bi)), ai, bi)[0]
        for ai, bi in zip(a, b)
    ]
)

# Display results
print(f"Resultado de la integral con MIS: {Imis}")
print(f"Resultado de la integral con integración numérica: {numerical_integral}")


# Define the different step sizes for trapezoidal rule
step_sizes = [0.001, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
results_table = []

# Convert Imis to scalar if it's an array
if isinstance(Imis, np.ndarray) and Imis.size == 1:
    Imis = float(Imis)

# Compute the integral using each step size and store in the results table
for h in step_sizes:
    integral = composite_trapezoidal(integrated_function, 0, 10, h)
    results_table.append([1 / h, h, integral, integral - Imis])


print("\nResults using composite trapezoidal method:")
print("Intervals | Step size | Integral | Difference from MIS")
print("-" * 60)
for row in results_table:
    print(f"{row[0]:>9.2f} | {row[1]:>9.3f} | {row[2]:>9.4f} | {row[3]:>18.4f}")


# Compute the function values for plotting
x_vals = np.linspace(0, 10, 1000)
y_vals = function(x_vals, a, b)

# Compute the PDF for plotting
pdf_vals = sum(
    [
        (1 / (sigma[i] * np.sqrt(2 * np.pi)))
        * np.exp(-((x_vals - mu[i]) ** 2) / (2 * sigma[i] ** 2))
        for i in range(3)
    ]
)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Function to be integrated", linewidth=2)
plt.plot(x_vals, pdf_vals, "k:", label="PDF")
plt.scatter(
    sampled_points_X,
    function(np.array(sampled_points_X), a, b),
    color="red",
    marker="*",
    s=10,
    label="Sampled Points (MIS)",
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function, PDF, and Sampled Points")
plt.legend()
plt.grid(True)
plt.show()

# Print the MIS result
print(f"Resultado de la integral con MIS: {np.mean(sampled_points_Y):.6f}")
