import numpy as np
import mpmath

def calculate_function_values(x, a, b, m, n):
    """Calculate the values of the function to be integrated."""
    total_sum = 0

    for i in range(m):
        products = [mpmath.sech(a[i] * (x[j] - b[i])) for j in range(n)]
        total_sum += np.prod(products)

    return total_sum

def calculate_pdf_i(x, a, b, n, index):
    """Calculate the probability density function (PDF)"""
    sigma = np.zeros(n)
    for i in range(n):
        sigma[i] = np.log(2 + np.sqrt(3)) / (a[i] * np.sqrt(2 * np.log(2)))

    products = [np.exp(-(x[j] - b[j]) ** 2 / (2 * sigma[j] ** 2)) for j in range(n)]
    return np.prod(products)

def calculate_exact_integral(a, b, m, n):
    """Calculate the exact integral of the function."""
    total_sum = 0

    for i in range(m):
        products = [a[i] for j in range(n)]
        total_sum += np.prod(products)

def calculate_balance_heuristic_weights(x, sample_counts, a, b, index, n):
    """Calculate weights using the balance heuristic method."""
    return (
        sample_counts[index]
        * calculate_pdf_i(x, a, b, n, index)
        / sum([
            sample_count * calculate_pdf_i(x, a, b, n, i)
            for i, sample_count in enumerate(sample_counts)
        ])
    )

def calculate_mis_estimate(total_samples, a, b, m, n):
    """Calculate the MIS estimate."""
    num_distributions = len(a)

    samples_per_distribution = np.round(a * total_samples).astype(int)
    total_samples = sum(samples_per_distribution)

    estimate = 0
    sampled_points_x = []
    sampled_points_y = []
    variance = 0
    iteration = 1
    t = 0

    for i in range(m):
        for j in range(n):
            x_sample = np.random.normal(b[i], a[i], n)
            y_sample = calculate_function_values(x_sample, a, b, m, n)

            weight = calculate_balance_heuristic_weights(x_sample, samples_per_distribution, a, b, i, num_distributions)

            sampled_points_x.append(x_sample)
            sampled_points_y.append(y_sample)

            weighted_sample = (float(weight * (y_sample / calculate_pdf_i(x_sample, a, b, num_distributions, i))) / samples_per_distribution[i]) * total_samples
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
    a = np.array([1.0, 1.0])
    b = np.array([0.0, 0.0])
    m = len(a)
    n = 5

    mis_estimate, sampled_points_x, _, variance, alternate_variance = calculate_mis_estimate(
        NUM_SAMPLES, a, b, m, n
    )

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Alternate variance of the integral with MIS: {alternate_variance}")
    print(f"Exact result of the integral: {calculate_exact_integral(a, b, m, n)}")

if __name__ == "__main__":
    main()
