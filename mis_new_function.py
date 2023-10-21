import numpy as np
import matplotlib.pyplot as plt

def calculate_function_values(x_values, lower_bounds, upper_bounds):
    x_values = np.atleast_1d(x_values)
    return np.exp(-x_values**2)

def calculate_normal_pdf(x, mean, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std_dev**2))

def calculate_balance_heuristic_weights(x, sample_counts, means, std_devs, index):
    return (
        sample_counts[index]
        * calculate_normal_pdf(x, means[index], std_devs[index])
        / sum([
            sample_count * calculate_normal_pdf(x, mean, std_dev)
            for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
        ])
    )

def calculate_power_heuristic_weights(x, sample_counts, means, std_devs, index, beta=2):
    pdf_values = [
        sample_count * calculate_normal_pdf(x, mean, std_dev)
        for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
    ]
    numerator = pdf_values[index] ** beta
    denominator = sum(pdf_val ** beta for pdf_val in pdf_values)
    return numerator / denominator

def calculate_mis_estimate(total_samples, alpha_values, means, std_devs, lower_bounds, upper_bounds, heuristic="balance"):
    num_distributions = len(alpha_values)
    samples_per_distribution = np.round(alpha_values * total_samples).astype(int)
    total_samples = sum(samples_per_distribution)
    estimate = 0
    sampled_points_x = []
    variance = 0
    iteration = 1
    t = 0
    for index in range(num_distributions):
        for _ in range(samples_per_distribution[index]):
            x_sample = std_devs[index] * np.random.randn() + means[index]
            y_sample = calculate_function_values(x_sample, lower_bounds, upper_bounds)
            weight = globals()[f"calculate_{heuristic}_heuristic_weights"](x_sample, samples_per_distribution, means, std_devs, index)
            sampled_points_x.append(x_sample)
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
    return estimate, sampled_points_x, variance, alternate_variance

def plot_results(x_values, y_values, pdf_values, sampled_points_x, lower_bounds, upper_bounds):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Function to be Integrated", color="blue", linewidth=2)
    plt.plot(x_values, pdf_values, label="PDF", color="green", linestyle="--", linewidth=2)
    for x in sampled_points_x:
        plt.plot(x, 0, 'ro')
    plt.axvline(x=lower_bounds[0], color='grey', linestyle='--')
    plt.axvline(x=upper_bounds[0], color='grey', linestyle='--')
    plt.title("MIS Estimation Visualization")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    NUM_SAMPLES = 50
    alpha_values = np.array([1.0])
    means = np.array([0.0])
    std_devs = np.array([1.0])
    lower_bounds = means - 2 * std_devs
    upper_bounds = means + 2 * std_devs
    mis_estimate, sampled_points_x, variance, alternate_variance = calculate_mis_estimate(
        NUM_SAMPLES, alpha_values, means, std_devs, lower_bounds, upper_bounds
    )
    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Alternate variance of the integral with MIS: {alternate_variance}")
    x_values = np.linspace(-3, 3, 1000)
    y_values = calculate_function_values(x_values, lower_bounds, upper_bounds)
    pdf_values = sum([
        (1 / (std_dev * np.sqrt(2 * np.pi)))
        * np.exp(-((x_values - mean) ** 2) / (2 * std_dev ** 2))
        for mean, std_dev in zip(means, std_devs)
    ])
    plot_results(x_values, y_values, pdf_values, sampled_points_x, lower_bounds, upper_bounds)

if __name__ == "__main__":
    main()