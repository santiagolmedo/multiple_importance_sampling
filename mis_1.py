import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import json
import time


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
    heuristic="balance",
):
    """Calculate the MIS estimate."""
    time_start = time.time()
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

    time_end = time.time()
    return (
        estimate,
        sampled_points_x,
        sampled_points_y,
        variance,
        alternate_variance,
        time_end - time_start,
    )


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


def run_mis_analysis():
    NUM_SAMPLES = [10, 25, 50, 100, 150]
    NUM_TESTS = 10
    NUM_RUNS = 1000
    heuristics = ["balance", "power", "maximum", "cutoff", "sbert"]
    general_results = []

    for test in range(NUM_TESTS):
        results = {
            heuristic: {num_samples: [] for num_samples in NUM_SAMPLES}
            for heuristic in heuristics
        }

        alpha_values = np.array([0.3333, 0.3333, 0.3333])
        means = np.sort(
            np.array(
                [
                    np.random.uniform(2, 7),
                    np.random.uniform(7, 12),
                    np.random.uniform(12, 18),
                ]
            )
        )
        std_devs = np.array(np.random.uniform(0.01, 1, 3))
        lower_bounds = means - 2 * std_devs
        upper_bounds = means + 2 * std_devs

        quad_result, quad_error = quad(
            lambda x: calculate_function_values(x, lower_bounds, upper_bounds),
            np.min(lower_bounds),
            np.max(upper_bounds),
        )

        for heuristic in heuristics:
            for num_samples in NUM_SAMPLES:
                for _ in range(NUM_RUNS):
                    (
                        estimate,
                        _,
                        _,
                        variance,
                        alt_variance,
                        time_taken,
                    ) = calculate_mis_estimate(
                        num_samples,
                        alpha_values,
                        means,
                        std_devs,
                        lower_bounds,
                        upper_bounds,
                        heuristic=heuristic,
                    )

                    standard_deviation = np.sqrt(variance)
                    error_compared_to_quad = np.abs(estimate - quad_result)

                    results[heuristic][num_samples].append(
                        (
                            estimate,
                            variance,
                            alt_variance,
                            standard_deviation,
                            error_compared_to_quad,
                            time_taken,
                        )
                    )

        general_results.append(
            [
                results,
                means,
                std_devs,
                lower_bounds,
                upper_bounds,
                quad_result,
                quad_error,
            ]
        )

    return general_results


def analyze_results(results):
    """Analyze the results of the MIS estimate."""

    analysis = {}

    for num_samples, data in results.items():
        (
            estimates,
            variances,
            alt_variances,
            standard_deviations,
            errors,
            time_taken,
        ) = zip(*data)
        analysis[num_samples] = {
            "min estimate": np.min(estimates),
            "max estimate": np.max(estimates),
            "mean of mis estimate": np.mean(estimates),
            "variance of estimates": np.var(estimates),
            "mean of variances": np.mean(variances),
            "mean of alternate variances": np.mean(alt_variances),
            "mean of standard deviations": np.mean(standard_deviations),
            "mean of errors": np.mean(errors),
            "mean time taken": "{:.4f}".format(np.mean(time_taken)) + " seconds",
        }

    return analysis


def print_analysis():
    # Test the modified functions
    mis_results = run_mis_analysis()

    analysis = {}

    for iter, test_data in enumerate(mis_results):
        analysis["Test " + str(iter + 1)] = {
            "test_values": {
                "means": test_data[1].tolist(),
                "std_devs": test_data[2].tolist(),
                "lower_bounds": test_data[3].tolist(),
                "upper_bounds": test_data[4].tolist(),
                "quad_result": test_data[5],
            }
        }
        for heuristic, results in test_data[0].items():
            analysis["Test " + str(iter + 1)][heuristic] = analyze_results(results)

    open("results_mis_1.txt", "w").close()

    with open("results_mis_1.txt", "w") as f:
        json.dump(analysis, f, indent=4)


def run_mis_estimate():
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 50
    alpha_values = np.array([0.3333, 0.3333, 0.3333])
    # means = np.sort(
    #     np.array(
    #         [
    #             np.random.uniform(2, 7),
    #             np.random.uniform(7, 12),
    #             np.random.uniform(12, 18),
    #         ]
    #     )
    # )
    means = np.array([5, 10, 15])
    std_devs = np.array([1, 0.5, 0.75])
    lower_bounds = means - 2 * std_devs
    upper_bounds = means + 2 * std_devs

    print(f"Means: {means}")
    print(f"Standard Deviations: {std_devs}")
    print(f"Lower Bounds: {lower_bounds}")
    print(f"Upper Bounds: {upper_bounds}")

    (
        mis_estimate,
        sampled_points_x,
        _,
        variance,
        alternate_variance,
        time_taken,
    ) = calculate_mis_estimate(
        NUM_SAMPLES, alpha_values, means, std_devs, lower_bounds, upper_bounds
    )

    result, error = quad(
        lambda x: calculate_function_values(x, lower_bounds, upper_bounds),
        np.min(lower_bounds),
        np.max(upper_bounds),
    )

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Result of the integral with quad: {result}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Alternate variance of the integral with MIS: {alternate_variance}")
    print(f"Error of the integral with quad: {error}")
    print(f"Time taken for MIS: {time_taken}")

    x_values = np.linspace(0, 20, 1000)
    y_values = calculate_function_values(x_values, lower_bounds, upper_bounds)
    pdf_values = sum(
        [
            (1 / (std_dev * np.sqrt(2 * np.pi)))
            * np.exp(-((x_values - mean) ** 2) / (2 * std_dev**2))
            for mean, std_dev in zip(means, std_devs)
        ]
    )

    plot_results(
        x_values, y_values, pdf_values, sampled_points_x, lower_bounds, upper_bounds
    )


if __name__ == "__main__":
    run_mis_estimate()
    # print_analysis()
