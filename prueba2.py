import numpy as np
import mpmath
import pdb

def calculate_function_values(x, a, b, m, n):
    """Calculate the values of the function to be integrated."""
    total_sum = 0
    for i in range(m):
        products = [mpmath.sech(a[i][j] * (x[j] - b[i][j])) for j in range(n)]
        total_sum += np.prod(products)

    return total_sum

def calculate_pdf_i(x, a, b, m, n, index):
    """Calculate the probability density function (PDF)"""
    sigma = np.zeros((m, n))
    for j in range(n):
      sigma[index][j] = np.log(2 + np.sqrt(3)) / (a[index][j] * np.sqrt(2 * np.log(2)))

    products = [np.exp(-(x[j] - b[index][j]) ** 2 / (2 * sigma[index][j] ** 2)) for j in range(n)]
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

    samples_per_distribution = [total_samples / num_distributions] * num_distributions
    for i in range(num_distributions):
        samples_per_distribution[i] = int(np.ceil(samples_per_distribution[i]))
    total_samples = sum(samples_per_distribution)

    estimate = 0
    sampled_points_x = []
    sampled_points_y = []
    variance = 0
    iteration = 1
    t = 0
    for i in range(num_distributions):
        for j in range(samples_per_distribution[i]):
            x_sample = np.array([np.random.normal(b[i, j], a[i, j]) for i in range(m) for j in range(n)])
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

def main():
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 50
    a = np.array([[1], [1], [1]])
    b = np.array([[0], [0], [0]])
    m = len(a)
    n = len(a[0])

    mis_estimate, sampled_points_x, _, variance, alternate_variance = calculate_mis_estimate(
        NUM_SAMPLES, a, b, m, n
    )

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Alternate variance of the integral with MIS: {alternate_variance}")
    print(f"Exact result of the integral: {calculate_exact_integral(a, m, n)}")

if __name__ == "__main__":
    main()

# Required imports for plotting
import matplotlib.pyplot as plt

# Enhancing the visualization of MIS samples
plt.figure(figsize=(14, 7))
plt.plot(x_values, function_values_uni, label="Function f(x)", color='blue')
plt.plot(x_values, pdf_values_uni, label="PDF", color='red', linestyle="--")
if mis_samples:
    mis_samples_y = [calculate_function_values_uni([x], a_uni, b_uni, m) for x in mis_samples]
    plt.scatter(mis_samples, mis_samples_y, color='green', s=200, label="MIS Samples", alpha=0.6, edgecolor='black', linewidth=1.5)
plt.title("Function f(x), PDF, and MIS Samples over [-10, 10]")
plt.xlabel("x")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# Comparing the histogram of the MIS samples with the actual PDF
plt.figure(figsize=(14, 7))
plt.hist(mis_samples, bins=50, density=True, alpha=0.6, color='green', label="Histogram of MIS Samples")
plt.plot(x_values, pdf_values_uni, label="PDF", color='red', linestyle="--", linewidth=2)
plt.title("Histogram of MIS Samples vs. PDF over [-10, 10]")
plt.xlabel("x")
plt.ylabel("Density/Value")
plt.legend()
plt.grid(True)
plt.show()