import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def calculate_function_values(x_values, lower_bounds, upper_bounds):
    """Calculate the values of the function to be integrated."""
    x_values = np.atleast_1d(x_values)
    return np.array(
        [
            np.sum(
                [
                    np.maximum(
                        0,
                        -(4 / (upper_bound - lower_bound) ** 2)
                        * (x - lower_bound)
                        * (x - upper_bound),
                    )
                    for lower_bound, upper_bound in zip(lower_bounds, upper_bounds)
                ]
            )
            for x in x_values
        ]
    )


def calculate_normal_pdf(x, mean, std_dev):
    """Calculate the probability density function (PDF) for a normal distribution."""
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -((x - mean) ** 2) / (2 * std_dev**2)
    )


def calculate_balance_heuristic_weights(x, sample_counts, means, std_devs, index):
    """Calculate weights using the balance heuristic method."""
    return (
        sample_counts[index]
        * calculate_normal_pdf(x, means[index], std_devs[index])
        / sum(
            [
                sample_count * calculate_normal_pdf(x, mean, std_dev)
                for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
            ]
        )
    )


def calculate_power_heuristic_weights(x, sample_counts, means, std_devs, index, beta=2):
    """Calculate weights using the power heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, mean, std_dev)
        for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
    ]

    numerator = pdf_values[index] ** beta
    denominator = sum(pdf_val**beta for pdf_val in pdf_values)

    return numerator / denominator


def calculate_maximum_heuristic_weights(x, sample_counts, means, std_devs, index):
    """Calculate weights using the maximum heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, mean, std_dev)
        for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
    ]

    return float(pdf_values[index] == max(pdf_values))


def calculate_cutoff_heuristic_weights(
    x, sample_counts, means, std_devs, index, alpha=0.5
):
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
    return calculate_normal_pdf(x, means[index], std_devs[index]) / sum(
        [
            calculate_normal_pdf(x, mean, std_dev)
            for sample_count, mean, std_dev in zip(sample_counts, means, std_devs)
        ]
    )


def calculate_mis_estimate(
    total_samples,
    alpha_values,
    means,
    std_devs,
    lower_bounds,
    upper_bounds,
    heuristic="sbert",
):
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
            x_sample = np.random.normal(means[index], std_devs[index])
            y_sample = calculate_function_values(x_sample, lower_bounds, upper_bounds)

            weight = globals()[f"calculate_{heuristic}_heuristic_weights"](
                x_sample, samples_per_distribution, means, std_devs, index
            )

            sampled_points_x.append(x_sample)
            sampled_points_y.append(y_sample)

            weighted_sample = (
                float(
                    weight
                    * (
                        y_sample
                        / calculate_normal_pdf(x_sample, means[index], std_devs[index])
                    )
                )
                / samples_per_distribution[index]
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


def calculate_basic_monte_carlo_estimate(
    total_samples,
    lower_bounds,
    upper_bounds,
):
    s = 0
    t = 0

    sampled_points_x = []
    sampled_points_y = []
    for iter in range(total_samples):
        sample = np.random.uniform(np.min(lower_bounds), np.max(upper_bounds))
        y = float(calculate_function_values(sample, lower_bounds, upper_bounds))

        if iter > 1:
            t += (1 - (1 / iter)) * (((y - s) / (iter - 1)) ** 2)
        s += y

        sampled_points_x.append(sample)
        sampled_points_y.append(y)

    estimate = s / total_samples
    sigma_variance = t / (total_samples - 1)
    variance = sigma_variance / total_samples

    return estimate, variance, sampled_points_x, sampled_points_y



def plot_results(
    x_values, y_values, pdf_values, sampled_points_x, lower_bounds, upper_bounds
):
    """Plot the results."""
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Function to be integrated", linewidth=2)
    plt.plot(x_values, pdf_values, "k:", label="PDF")
    plt.scatter(
        sampled_points_x,
        calculate_function_values(
            np.array(sampled_points_x), lower_bounds, upper_bounds
        ),
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


def run_mis_analysis(
    num_runs=1000,
    heuristics=["balance", "power", "maximum", "cutoff", "sbert"],
    num_samples_list=[10, 25, 50, 100],
):
    results = {
        heuristic: {num_samples: [] for num_samples in num_samples_list}
        for heuristic in heuristics
    }

    alpha_values = np.array([0.3333, 0.3333, 0.3333])
    means = np.array([2, 5, 7])
    std_devs = np.array([0.8, 0.8, 0.4]) / 2
    lower_bounds = means - 2 * std_devs
    upper_bounds = means + 2 * std_devs

    for heuristic in heuristics:
        for num_samples in num_samples_list:
            for _ in range(num_runs):
                estimate, _, _, variance, alt_variance = calculate_mis_estimate(
                    num_samples,
                    alpha_values,
                    means,
                    std_devs,
                    lower_bounds,
                    upper_bounds,
                    heuristic=heuristic,
                )
                results[heuristic][num_samples].append(
                    (estimate, variance, alt_variance)
                )

    return results


def analyze_results(results):
    analysis = {}

    for heuristic, data in results.items():
        estimates, variances, alt_variances = zip(*data)
        analysis[heuristic] = {
            "min_estimate": np.min(estimates),
            "max_estimate": np.max(estimates),
            "mean_estimate": np.mean(estimates),
            "variance_of_estimates": np.var(estimates),
            "mean_of_variances": np.mean(variances),
            "mean_of_alternate_variances": np.mean(alt_variances),
        }

    return analysis


def print_analysis():
    # Test the modified functions
    mis_results = run_mis_analysis()
    analysis = {
        heuristic: {
            num_samples: analyze_results({heuristic: data})[heuristic]
            for num_samples, data in sample_data.items()
        }
        for heuristic, sample_data in mis_results.items()
    }
    print(analysis)


def run_mis_estimate():
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 50000
    alpha_values = np.array([0.3333, 0.3333, 0.3333])
    means = np.array([2, 5, 7])
    std_devs = np.array([0.8, 0.8, 0.4]) / 2
    lower_bounds = means - 2 * std_devs
    upper_bounds = means + 2 * std_devs

    # (
    #     mis_estimate,
    #     sampled_points_x,
    #     _,
    #     variance,
    #     alternate_variance,
    # ) = calculate_mis_estimate(
    #     NUM_SAMPLES, alpha_values, means, std_devs, lower_bounds, upper_bounds
    # )

    (
        basic_mc_estimate,
        basic_mc_variance,
        basic_mc_sampled_points_x,
        basic_mc_sampled_points_y,
    ) = calculate_basic_monte_carlo_estimate(
        NUM_SAMPLES, lower_bounds, upper_bounds
    )

    result, error = quad(
        lambda x: calculate_function_values(x, lower_bounds, upper_bounds),
        np.min(lower_bounds),
        np.max(upper_bounds),
    )

    # print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Result of the integral with basic MC: {basic_mc_estimate}")
    print(f"Result of the integral with quad: {result}")
    # print(f"Variance of the integral with MIS: {variance}")
    # print(f"Alternate variance of the integral with MIS: {alternate_variance}")
    print(f"Variance of the integral with basic MC: {basic_mc_variance}")
    print(f"Error of the integral with quad: {error}")

    x_values = np.linspace(0, 10, 1000)
    y_values = calculate_function_values(x_values, lower_bounds, upper_bounds)
    pdf_values = sum(
        [
            (1 / (std_dev * np.sqrt(2 * np.pi)))
            * np.exp(-((x_values - mean) ** 2) / (2 * std_dev**2))
            for mean, std_dev in zip(means, std_devs)
        ]
    )

    # plot_results(
    #     x_values, y_values, pdf_values, sampled_points_x, lower_bounds, upper_bounds
    # )

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Function to be integrated", linewidth=2)
    plt.scatter(basic_mc_sampled_points_x, basic_mc_sampled_points_y, color="red", marker="*", s=100, label="Sampled Points (Basic MC)")
    plt.show()


if __name__ == "__main__":
    run_mis_estimate()
    # print_analysis()
