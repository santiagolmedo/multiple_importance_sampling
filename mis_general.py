import numpy as np
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

def compute_balance_heuristic_weights(x_ij, ni, mu, sigma, k, i, pi_x):
    """Compute the balance heuristic weights."""
    return (ni[i] * pi_x) / sum(compute_p_k(x_ij, mu[k_iterator], sigma[k_iterator]) for k_iterator in range(k))



def compute_mis_estimate(n, alfai, mu, sigma, a, b):
    """Compute the MIS estimate."""
    K = len(alfai)
    ni = np.round(alfai * n).astype(int)
    n = sum(ni)
    F = 0
    sampled_points_X = []

    variance = 0
    m = 1
    t = 0
    for i in range(K):
        for j in range(ni[i]):
            x_ij = sigma[i] * np.random.randn() + mu[i]
            sampled_points_X.append(x_ij)

            Y = compute_function_values(x_ij, a, b)
            pi_x = compute_p_k(x_ij, mu[i], sigma[i])
            weights = compute_balance_heuristic_weights(x_ij, ni, mu, sigma, K, i, pi_x)

            x = (float(Y / pi_x) * weights) / ni[i]

            if m > 1:
              t += (1 - (1 / m)) * ((x - F / (m - 1)) ** 2)

            F += x
            variance += x**2
            m += 1

    F = F / n
    variance = (variance / (n**2 - n)) - (F**2 / (n - 1))

    sigma = t / (n - 1)
    variance_alt = sigma / n

    return F, variance, variance_alt, sampled_points_X


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
    Imis, variance, variance_alt, sampled_points_X = compute_mis_estimate(
        N, alfai, mu, sigma, a, b
    )

    # Display results
    print(f"Resultado de la integral con MIS: {Imis}")
    print(f"Varianza 1 de la integral con MIS: {variance}")
    print(f"Varianza 2 de la integral con MIS: {variance_alt}")

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
if __name__ == "__main__":
    main()
