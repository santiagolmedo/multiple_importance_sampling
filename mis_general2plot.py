import numpy as np
import mpmath
import matplotlib.pyplot as plt

def calculate_function_values(x, a, b, m, n):
    """Calculate the values of the function to be integrated."""
    total_sum = 0

    for i in range(m):
        products = [mpmath.sech(a[i][j] * (x[i][j] - b[i][j])) for j in range(n)]
        total_sum += np.prod(products)

    return total_sum

def calculate_pdf_i(x, a, b, m, n, index):
    """Calculate the probability density function (PDF)"""
    sigma = np.zeros((n, m))
    for j in range(n):
      sigma[index][j] = np.log(2 + np.sqrt(3)) / (a[index][j] * np.sqrt(2 * np.log(2)))

    products = [np.exp(-(x[index][j] - b[index][j]) ** 2 / (2 * sigma[index][j] ** 2)) for j in range(n)]
    return np.prod(products)

def calculate_exact_integral(a, m, n):
    """Calculate the exact integral of the function."""
    total_sum = 0

    for i in range(m):
        products = [a[i][j] for j in range(n)]
        total_sum += (np.pi ** n) / np.prod(products)

    return total_sum

def calculate_balance_heuristic_weights(x, sample_counts, a, b, index, m, n):
    """Calculate weights using the balance heuristic method."""
    return (
        sample_counts[index]
        * calculate_pdf_i(x, a, b, m, n, index)
        / sum([
            sample_count * calculate_pdf_i(x, a, b, m, n, i)
            for i, sample_count in enumerate(sample_counts)
        ])
    )

def calculate_mis_estimate(total_samples, a, b, m, n):
    """Calculate the MIS estimate."""
    num_distributions = m

    samples_per_distribution = [total_samples // num_distributions] * num_distributions
    samples_per_distribution[0] += total_samples % num_distributions
    total_samples = sum(samples_per_distribution)

    estimate = 0
    sampled_points_x = []
    sampled_points_y = []
    variance = 0
    iteration = 1
    t = 0

    for i in range(m):
        for j in range(n):
            x_sample = np.random.normal(b[i][j], a[i][j], (m,n))
            y_sample = calculate_function_values(x_sample, a, b, m, n)

            weight = calculate_balance_heuristic_weights(x_sample, samples_per_distribution, a, b, i, m, n)

            sampled_points_x.append(x_sample)
            sampled_points_y.append(y_sample)

            weighted_sample = (float(weight * (y_sample / calculate_pdf_i(x_sample, a, b, m, n, i))) / samples_per_distribution[i]) * total_samples
            if iteration > 1:
                t += (1 - (1 / iteration)) * ((weighted_sample - estimate / (iteration - 1)) ** 2)

            estimate += weighted_sample
            variance += weighted_sample**2
            iteration += 1

    estimate = estimate / total_samples
    variance = (variance / (total_samples**2 - total_samples)) - (estimate**2 / (total_samples - 1))
    sigma_variance = t / (total_samples - 1)
    alternate_variance = sigma_variance / total_samples

    return estimate, sampled_points_x, sampled_points_y, variance, alternate_variance

# Function for 2D visualization

def plot_2d_results(a, b, m, n):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)

    X, Y = np.meshgrid(x, y)
    Z_func = np.zeros_like(X)
    Z_pdf = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            input_val = np.array([[X[i, j], Y[i, j]] for _ in range(m)])
            Z_func[i, j] = calculate_function_values(input_val, a, b, m, n)
            Z_pdf[i, j] = calculate_pdf_i(input_val, a, b, m, n, 0)  # Using first distribution as an example

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting the function
    c1 = ax1.contourf(X, Y, Z_func, cmap='viridis')
    ax1.set_title('Function to be Integrated')
    fig.colorbar(c1, ax=ax1)

    # Plotting the PDF
    c2 = ax2.contourf(X, Y, Z_pdf, cmap='viridis')
    ax2.set_title('Probability Density Function (PDF)')
    fig.colorbar(c2, ax=ax2)

    plt.tight_layout()
    plt.show()

# Function for 3D side view visualization

def plot_3d_side_view(a, b, m, n):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)

    X, Y = np.meshgrid(x, y)
    Z_func = np.zeros_like(X)
    Z_pdf = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            input_val = np.array([[X[i, j], Y[i, j]] for _ in range(m)])
            Z_func[i, j] = calculate_function_values(input_val, a, b, m, n)
            Z_pdf[i, j] = calculate_pdf_i(input_val, a, b, m, n, 0)  # Using first distribution as an example

    fig = plt.figure(figsize=(14, 6))

    # Plotting the function in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z_func, cmap='viridis')
    ax1.view_init(elev=0, azim=-90)  # side view
    ax1.set_title('Function to be Integrated')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plotting the PDF in 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, Z_pdf, cmap='viridis')
    ax2.view_init(elev=0, azim=-90)  # side view
    ax2.set_title('Probability Density Function (PDF)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

def main():
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 50
    a = np.array([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0]])
    m = 2
    n = 2

    mis_estimate, sampled_points_x, _, variance, alternate_variance = calculate_mis_estimate(
        NUM_SAMPLES, a, b, m, n
    )

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Alternate variance of the integral with MIS: {alternate_variance}")
    print(f"Exact result of the integral: {calculate_exact_integral(a, m, n)}")

    plot_2d_results(a, b, m, n)
    plot_3d_side_view(a, b, m, n)

if __name__ == "__main__":
    main()