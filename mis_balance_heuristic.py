import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


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


def compute_balance_heuristic_weights(X, ni, mu, sigma):
    """Compute the weights using the balance heuristic."""
    K = len(mu)
    pi_X_values = [compute_p_k(X, mu[k], sigma[k]) for k in range(K)]
    return [
        (ni[i] * pi_X_values[i]) / sum([(ni[k] * pi_X_values[k]) for k in range(K)])
        for i in range(K)
    ]


def compute_power_heuristic_weights(X, ni, mu, sigma, beta=2):
    """Compute the weights using the power heuristic."""
    K = len(mu)
    pi_X_values = [compute_p_k(X, mu[k], sigma[k]) for k in range(K)]
    return [
        (ni[i] * pi_X_values[i]) ** beta
        / sum([(ni[k] * pi_X_values[k]) ** beta for k in range(K)])
        for i in range(K)
    ]


def compute_cutoff_heuristic_weights(X, ni, mu, sigma):
    """Compute the weights using the cutoff heuristic."""
    K = len(mu)
    pi_X_values = [compute_p_k(X, mu[k], sigma[k]) for k in range(K)]
    max_val_index = np.argmax([ni[k] * pi_X_values[k] for k in range(K)])
    return [1 if i == max_val_index else 0 for i in range(K)]


def compute_maximum_heuristic_weights(X, ni, mu, sigma):
    """Compute the weights using the maximum heuristic."""
    K = len(mu)
    pi_X_values = [compute_p_k(X, mu[k], sigma[k]) for k in range(K)]
    max_val = max([ni[k] * pi_X_values[k] for k in range(K)])
    return [(ni[i] * pi_X_values[i]) / max_val for i in range(K)]


def compute_mis_estimate(N, alfai, mu, sigma, a, b):
    """Compute the MIS estimate."""
    K = len(alfai)
    ni = np.round(alfai * N).astype(int)
    F = 0
    sampled_points_X = []
    sampled_points_Y = []

    accumulator1 = []
    accumulator2 = []

    for i in range(K):
        total_sum = 0
        for j in range(ni[i]):
            X = sigma[i] * np.random.randn() + mu[i]
            Y = compute_function_values(X, a, b)

            # Compute the weights
            weights = compute_balance_heuristic_weights(X, ni, mu, sigma)

            # Add the sampled point values to the lists
            sampled_points_X.append(X)
            sampled_points_Y.append(Y)

            total_sum += weights[i] * (Y / compute_p_k(X, mu[i], sigma[i]))

            accumulator1.append(((weights[i] ** 2) * (Y ** 2)) / (compute_p_k(X, mu[i], sigma[i]) * ni[i]))
            accumulator2.append((quad(lambda x: weights[i] * compute_function_values(x, a, b), a[i], b[i])[0] ** 2) / ni[i])

        F += total_sum / ni[i]

        variance = quad(lambda x: np.sum(accumulator1), min(a), max(b))[0] - sum(accumulator2)

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
        s=10,
        label="Sampled Points (MIS)",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function, PDF, and Sampled Points")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Initial parameters
    N = 50
    alfai = np.array([0.3333, 0.3333, 0.3333])
    mu = np.array([2, 5, 7])
    sigma = np.array([0.8, 0.8, 0.4]) / 2
    a = mu - 2 * sigma
    b = mu + 2 * sigma

    # Seed for reproducibility
    np.random.seed(8)

    # Compute MIS estimate
    Imis, sampled_points_X, _, variance = compute_mis_estimate(N, alfai, mu, sigma, a, b)

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
    x_vals = np.linspace(0, 10, 1000)
    y_vals = compute_function_values(x_vals, a, b)
    pdf_vals = sum(
        [
            (1 / (sigma[i] * np.sqrt(2 * np.pi)))
            * np.exp(-((x_vals - mu[i]) ** 2) / (2 * sigma[i] ** 2))
            for i in range(3)
        ]
    )

    # Plot the results
    plot_results(x_vals, y_vals, pdf_vals, sampled_points_X, a, b)


# Execute the main function
main()
