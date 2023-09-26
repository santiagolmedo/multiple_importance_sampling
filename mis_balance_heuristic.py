import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pdb


def compute_function_values(x, a, b):
    """Compute the function values for a given set of parameters."""
    x = np.atleast_1d(x)
    return np.array(
        [
            np.sum(
                [
                    np.maximum(0, -(4 / (b[i] - a[i]) ** 2) * (xi - a[i]) * (xi - b[i]))
                    for i in range(len(a))
                ]
            )
            for xi in x
        ]
    )


def compute_p_k(X, mu_k, sigma_k):
    """Compute the p_k value for a given set of parameters."""
    return (1 / (sigma_k * np.sqrt(2 * np.pi))) * np.exp(
        -((X - mu_k) ** 2) / (2 * sigma_k**2)
    )


def compute_composite_trapezoidal(f, a, b, h):
    """Compute the integral using the composite trapezoidal rule."""
    x = np.arange(a, b + h, h)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def compute_integrated_function(x, a, b):
    """Compute the integrated function values."""
    x = np.asarray(x).reshape(-1, 1)
    return np.sum(np.maximum(0, -(4 / (b - a) ** 2) * (x - a) * (x - b)), axis=1)


def compute_balance_heuristic_weights(X, ni, mu, sigma, i):
    """Compute the weights using the balance heuristic."""
    return (
        ni[i]
        * compute_p_k(X, mu[i], sigma[i])
        / sum([ni[k] * compute_p_k(X, mu[k], sigma[k]) for k in range(len(mu))])
    )


def compute_mis_estimate(N, alfai, mu, sigma, a, b):
    """Compute the MIS estimate."""
    K = len(alfai)
    ni = np.round(alfai * N).astype(int)
    F = 0
    sampled_points_X = []
    sampled_points_Y = []

    s = 0
    t = 0
    variance_f_estimate = 0
    variance = 0

    for i in range(K):
        total_sum = 0
        t = 0

        for j in range(ni[i]):
            X = sigma[i] * np.random.randn() + mu[i]
            Y = compute_function_values(X, a, b)

            # Compute the weights
            weight = compute_balance_heuristic_weights(X, ni, mu, sigma, i)

            # Add the sampled point values to the lists
            sampled_points_X.append(X)
            sampled_points_Y.append(Y)

            if j > 0:
                t += (1 - 1 / (j + 1)) * value_ij - total_sum / ((j) ** 2)

            value_ij = float(weight * (Y / compute_p_k(X, mu[i], sigma[i])))
            total_sum += value_ij

        F += total_sum / ni[i]
        variance_f_estimate += t / (ni[i] - 1)
        variance += variance_f_estimate / ni[i]

    return F, sampled_points_X, sampled_points_Y, variance


def compute_numerical_integral(a, b):
    """Compute the numerical integral."""
    return sum(
        [
            quad(lambda x: max(0, -(4 / (bi - ai) ** 2) * (x - ai) * (x - bi)), ai, bi)[
                0
            ]
            for ai, bi in zip(a, b)
        ]
    )


def compute_trapezoidal_results(Imis, a, b):
    """Compute the results using the composite trapezoidal method."""
    step_sizes = [0.001, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    results = []
    if isinstance(Imis, np.ndarray) and Imis.size == 1:
        Imis = float(Imis)
    for h in step_sizes:
        integral = compute_composite_trapezoidal(
            lambda x: compute_integrated_function(x, a, b), 0, 10, h
        )
        results.append([1 / h, h, integral, integral - Imis])
    return results


def plot_results(x_vals, y_vals, pdf_vals, sampled_points_X, a, b):
    """Plot the results."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="Function to be integrated", linewidth=2)
    plt.plot(x_vals, pdf_vals, "k:", label="PDF")
    plt.scatter(
        sampled_points_X,
        compute_function_values(np.array(sampled_points_X), a, b),
        color="red",
        marker="*",
        s=100,
        label="Sampled Points (MIS)",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function, PDF, and Sampled Points")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_mis_analysis(N, alfai, mu, sigma, a, b):
    Imis_array = []
    variance_array = []
    for i in range(1000):
        Imis, sampled_points_X, _, variance = compute_mis_estimate(
            N, alfai, mu, sigma, a, b
        )

        Imis_array.append(Imis)
        variance_array.append(variance)

    print(f"Promedio de la integral con MIS: {np.mean(Imis_array)}")
    print(f"Varianza de la integral con MIS: {np.var(Imis_array)}")
    print(f"Desviación estándar de la integral con MIS: {np.std(Imis_array)}")
    print(f"Error de la integral con MIS: {np.std(Imis_array) / np.sqrt(1000)}")
    print(f"Mínimo de la integral con MIS: {np.min(Imis_array)}")
    print(f"Máximo de la integral con MIS: {np.max(Imis_array)}")

    ## hacer calculos en base al variance_array
    print(f"Promedio de la varianza con MIS: {np.mean(variance_array)}")
    print(f"Mínimo de la varianza con MIS: {np.min(variance_array)}")
    print(f"Máximo de la varianza con MIS: {np.max(variance_array)}")


def main():
    # Initial parameters
    N = 50
    alfai = np.array([0.3333, 0.3333, 0.3333])
    mu = np.array([2, 5, 7])
    sigma = np.array([0.8, 0.8, 0.4]) / 2
    a = mu - 2 * sigma
    b = mu + 2 * sigma

    # Seed for reproducibility
    # np.random.seed(8)

    # Compute MIS estimate
    Imis, sampled_points_X, _, variance = compute_mis_estimate(
        N, alfai, mu, sigma, a, b
    )

    # Numerical integration
    numerical_integral = compute_numerical_integral(a, b)

    # Display results
    print(f"Resultado de la integral con MIS: {Imis}")
    print(f"Varianza de la integral con MIS: {variance}")
    print(f"Resultado de la integral con integración numérica: {numerical_integral}")

    # Compute the results using the trapezoidal method
    results_table = compute_trapezoidal_results(Imis, a, b)

    # Display the results
    print("\nResults using composite trapezoidal method:")
    print("Intervals | Step size | Integral | Difference from MIS")
    print("-" * 60)
    for row in results_table:
        print(f"{row[0]:>9.2f} | {row[1]:>9.3f} | {row[2]:>9.4f} | {row[3]:>18.4f}")

    # Compute the function values for plotting
    # x_vals = np.linspace(0, 10, 1000)
    # y_vals = compute_function_values(x_vals, a, b)
    # pdf_vals = sum(
    #     [
    #         (1 / (sigma[i] * np.sqrt(2 * np.pi)))
    #         * np.exp(-((x_vals - mu[i]) ** 2) / (2 * sigma[i] ** 2))
    #         for i in range(3)
    #     ]
    # )

    # Plot the results
    # plot_results(x_vals, y_vals, pdf_vals, sampled_points_X, a, b)


# Execute the main function
if __name__ == "__main__":
    main()
