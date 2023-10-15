import numpy as np
import matplotlib.pyplot as plt

def calculate_function_values(x_values, lower_bounds, upper_bounds):
    """Calculate the values of the function to be integrated."""
    x_values = np.atleast_1d(x_values)
    return np.array([
        np.sum([
            np.maximum(0, -(4 / (upper_bound - lower_bound) ** 2) * (x - lower_bound) * (x - upper_bound))
            for lower_bound, upper_bound in zip(lower_bounds, upper_bounds)
        ])
        for x in x_values
    ])

def calculate_normal_pdf(x, mean, std_dev):
    """Calculate the probability density function (PDF) for a normal distribution."""
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))

def calculate_balance_heuristic_weights(x, sample_counts, means, std_devs, index):
    """Calculate weights using the balance heuristic method."""
    return (
        sample_counts[index]
        * calculate_normal_pdf(x, means[index], std_devs[index])
        / sum([
            sample_count * calculate_normal_pdf(x, mean, std_dev)
            for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
        ])
    )

def calculate_power_heuristic_weights(x, sample_counts, means, std_devs, index, beta=2):
    """Calculate weights using the power heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, mean, std_dev)
        for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
    ]

    numerator = pdf_values[index] ** beta
    denominator = sum(pdf_val ** beta for pdf_val in pdf_values)

    return numerator / denominator

def calculate_maximum_heuristic_weights(x, sample_counts, means, std_devs, index):
    """Calculate weights using the maximum heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, mean, std_dev)
        for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
    ]

    return float(pdf_values[index] == max(pdf_values))

def calculate_cutoff_heuristic_weights(x, sample_counts, means, std_devs, index, alpha=0.5):
    """Calculate weights using the cutoff heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, mean, std_dev)
        for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
    ]

    q_max = max(pdf_values)
    q_index = pdf_values[index]

    if q_index < alpha * q_max:
        return 0
    else:
        denominator = sum(q_k for q_k in pdf_values if q_k >= alpha * q_max)
        return q_index / denominator

def calculate_sbert_heuristic_weights(x, sample_counts, means, std_devs, index):
    """Calculate weights using the SBERT method."""
    return (
        calculate_normal_pdf(x, means[index], std_devs[index])
        / sum([
            calculate_normal_pdf(x, mean, std_dev)
            for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
        ])
    )

def calculate_mis_estimate(total_samples, alpha_values, means, std_devs, lower_bounds, upper_bounds, heuristic="sbert"):
    """Calculate the MIS estimate."""
    num_distributions = len(alpha_values)

    samples_per_distribution = np.round(alpha_values * total_samples).astype(int)
    total_samples = sum(samples_per_distribution)

    estimate = 0
    sampled_points_x = []
    sampled_points_y = []
    variance = 0
    iteration = 1
    t = 0

    for index in range(num_distributions):
        for _ in range(samples_per_distribution[index]):
            x_sample = std_devs[index] * np.random.randn() + means[index]
            y_sample = calculate_function_values(x_sample, lower_bounds, upper_bounds)

            weight = globals()[f"calculate_{heuristic}_heuristic_weights"](x_sample, samples_per_distribution, means, std_devs, index)

            sampled_points_x.append(x_sample)
            sampled_points_y.append(y_sample)

            weighted_sample = (float(weight * (y_sample / calculate_normal_pdf(x_sample, means[index], std_devs[index]))) / samples_per_distribution[index]) * total_samples
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

def plot_results(x_values, y_values, pdf_values, sampled_points_x, lower_bounds, upper_bounds):
    """Plot the results."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Function to be integrated", linewidth=2)
    plt.plot(x_values, pdf_values, "k:", label="PDF")
    plt.scatter(
        sampled_points_x,
        calculate_function_values(np.array(sampled_points_x), lower_bounds, upper_bounds),
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
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 50
    alpha_values = np.array([0.3333, 0.3333, 0.3333])
    means = np.array([2, 5, 7])
    std_devs = np.array([0.8, 0.8, 0.4]) / 2
    lower_bounds = means - 2 * std_devs
    upper_bounds = means + 2 * std_devs

    np.random.seed(8)

    mis_estimate, sampled_points_x, _, variance, alternate_variance = calculate_mis_estimate(
        NUM_SAMPLES, alpha_values, means, std_devs, lower_bounds, upper_bounds
    )

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Alternate variance of the integral with MIS: {alternate_variance}")

    x_values = np.linspace(0, 10, 1000)
    y_values = calculate_function_values(x_values, lower_bounds, upper_bounds)
    pdf_values = sum([
        (1 / (std_dev * np.sqrt(2 * np.pi)))
        * np.exp(-((x_values - mean) ** 2) / (2 * std_dev ** 2))
        for mean, std_dev in zip(means, std_devs)
    ])

    plot_results(x_values, y_values, pdf_values, sampled_points_x, lower_bounds, upper_bounds)

if __name__ == "__main__":
    main()

