import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def calculate_function_values(x):
    f1 = np.maximum(0, -(x - 10) * (x + 10) / 1000)
    f2 = np.maximum(0, -(x - 0.01) * (x + 0.01) * 10000)

    return f1 * f2


def normal_pdf(x, mu=0, sigma=1):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def calculate_balance_heuristic_weights(x, sample_counts, mu, sigma, index):
    """Calculate weights using the balance heuristic method."""
    return (
        sample_counts[index]
        * normal_pdf(x, mu[index], sigma[index])
        / sum(
            [
                sample_count * normal_pdf(x, mean, std_dev)
                for sample_count, mean, std_dev in zip(sample_counts, mu, sigma)
            ]
        )
    )


def calculate_mis_estimate(
    total_samples, heuristic, mu_1=0, sigma_1=10, mu_2=0, sigma_2=0.01
):
    """Calculate the MIS estimate."""

    num_distributions = 2
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
            mu = mu_1 if i == 0 else mu_2
            sigma = sigma_1 if i == 0 else sigma_2
            x_sample = np.random.normal(mu, sigma)
            y_sample = calculate_function_values(x_sample)
            weight = globals()[f"calculate_{heuristic}_heuristic_weights"](
                x_sample, samples_per_distribution, [mu_1, mu_2], [sigma_1, sigma_2], i
            )

            sampled_points_x.append(x_sample)
            sampled_points_y.append(y_sample)

            weighted_sample = (
                float(weight * (y_sample / normal_pdf(x_sample, mu, sigma)))
                / samples_per_distribution[i]
            ) * total_samples
            if iteration > 1:
                t += (1 - (1 / iteration)) * (
                    (weighted_sample - estimate / (iteration - 1)) ** 2
                )

            estimate += weighted_sample
            variance += weighted_sample**2
            iteration += 1

    estimate = estimate / total_samples
    variance = (variance / (total_samples**2 - total_samples)) - (
        estimate**2 / (total_samples - 1)
    )
    sigma_variance = t / (total_samples - 1)
    alternate_variance = sigma_variance / total_samples

    return estimate, sampled_points_x, sampled_points_y, variance, alternate_variance


def main():
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 100
    mu_1 = 0
    sigma_1 = 10
    mu_2 = 0
    sigma_2 = 0.01

    (
        estimate,
        sampled_points_x,
        sampled_points_y,
        variance,
        alternate_variance,
    ) = calculate_mis_estimate(NUM_SAMPLES, "balance", mu_1, sigma_1, mu_2, sigma_2)

    integral_value = quad(calculate_function_values, -np.inf, np.inf)

    print(f"Integral: {integral_value[0]}")

    print(f"Estimate: {estimate}")
    print(f"Variance: {variance}")
    print(f"Alternate Variance: {alternate_variance}")
    print(f"Error: {np.abs(integral_value[0] - estimate)}")


if __name__ == "__main__":
    main()
