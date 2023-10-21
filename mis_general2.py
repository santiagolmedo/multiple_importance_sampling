import numpy as np
import matplotlib.pyplot as plt
import mpmath

def calculate_function_values(x_value, a, c, n):
    """Calculate the values of the function to be integrated."""
    products = [mpmath.sech(a[i] * (x_value[i] - c[i])) for i in range(n)]
    return np.prod(products)

def calculate_pdf(x_value, a, c, n):
    """Calculate the probability density function (PDF)"""
    sigma = np.zeros(n)
    for i in range(n):
        sigma[i] = (np.log(2 + np.sqrt(3)) / a[i]) * np.sqrt(1 / 2 * np.log(2))

    return (1 / ((2 * np.pi) ** (n/2)) * np.prod(sigma)) * np.exp((-1/2) * sum([((x_value[i] - c[i]) / sigma[i]) ** 2 for i in range(n)]))

def calculate_balance_heuristic_weights(x, sample_counts, a, c, index, n):
    """Calculate weights using the balance heuristic method."""
    return (
        sample_counts[index]
        * calculate_pdf(x[index], a[index], c[index], n)
        / sum([
            sample_count * calculate_pdf(x[i], a[i], c[i], n)
            for i, sample_count in enumerate(sample_counts)
        ])
    )

def calculate_mis_estimate(total_samples, a, c, heuristic="balance"):
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

    for index in range(num_distributions):
        for _ in range(samples_per_distribution[index]):
            x_sample = np.random.uniform(0, 1, num_distributions)
            y_sample = calculate_function_values(x_sample, a, c, num_distributions)

            weight = calculate_balance_heuristic_weights(x_sample, samples_per_distribution, a, c, index, num_distributions)

            sampled_points_x.append(x_sample)
            sampled_points_y.append(y_sample)

            weighted_sample = (float(weight * (y_sample / calculate_pdf(x_sample, a[index], c[index], num_distributions))) / samples_per_distribution[index]) * total_samples
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
    c = np.array([0.0, 0.0])

    mis_estimate, sampled_points_x, _, variance, alternate_variance = calculate_mis_estimate(
        NUM_SAMPLES, a, c
    )

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Alternate variance of the integral with MIS: {alternate_variance}")

if __name__ == "__main__":
    main()
