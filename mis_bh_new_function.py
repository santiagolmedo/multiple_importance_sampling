import numpy as np
import matplotlib.pyplot as plt

def compute_function_values(x, a, b):
    """Compute the function values for a given set of parameters."""
    x = np.atleast_1d(x)
    return np.sin(a * x) * np.exp(-b * x)

def compute_p_k(X, a, b):
    """Compute the p_k value for a given set of parameters."""
    # Example: Exponential distribution
    return b * np.exp(-b * X)

def compute_mis_estimate(n, a, b):
    """Compute the MIS estimate."""
    F = 0
    sampled_points_X = []

    variance = 0
    m = 1
    t = 0
    for j in range(n):
        X = -np.log(1 - np.random.rand()) / b  # Inverse transform sampling for exponential
        sampled_points_X.append(X)

        Y = compute_function_values(X, a, b)
        p_techo = compute_p_k(X, a, b)
        x = float(Y / p_techo)

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
    a = 2  # Frequency parameter for sin function
    b = 0.5  # Decay parameter for exponential function

    # Seed for reproducibility
    # np.random.seed(8)

    # Compute MIS estimate
    Imis, variance, variance_alt, sampled_points_X = compute_mis_estimate(N, a, b)

    # Display results
    print(f"Resultado de la integral con MIS: {Imis}")
    print(f"Varianza 1 de la integral con MIS: {variance}")
    print(f"Varianza 2 de la integral con MIS: {variance_alt}")

    # Compute the function values for plotting
    x_vals = np.linspace(0, 10, 1000)
    y_vals = compute_function_values(x_vals, a, b)
    pdf_vals = compute_p_k(x_vals, a, b)

    # Plot the results
    plot_results(x_vals, y_vals, pdf_vals, sampled_points_X, a, b)

# Execute the main function
if __name__ == "__main__":
    main()
