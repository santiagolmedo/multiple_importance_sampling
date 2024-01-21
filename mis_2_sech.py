import numpy as np
import matplotlib.pyplot as plt
import time
import json


def calculate_function_values(x, a, b, m, n):
    """Optimized function to calculate the values."""
    x_matrix = x.reshape(1, n)
    sech_matrix = 1 / np.cosh(a * (x_matrix - b))
    total_sum = np.prod(sech_matrix, axis=1).sum()
    return float(total_sum)


def calculate_sigma(a, index):
    sigma = np.log(2 + np.sqrt(3)) / (a[index] * np.sqrt(2 * np.log(2)))
    return sigma


def normal_pdf(x, mu=0, sigma=1):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def calculate_normal_pdf(x, a, b, sigma, m, n, index):
    products = [
        normal_pdf(x[j], mu=b[index][j], sigma=sigma[index][j]) for j in range(n)
    ]
    return np.prod(products)


def calculate_exact_integral(a, m, n):
    """Calculate the exact integral of the function."""
    total_sum = 0

    for i in range(m):
        products = [a[i][j] for j in range(n)]
        total_sum += (np.pi**n) / np.prod(products)

    return total_sum


def calculate_balance_heuristic_weights(x, sample_counts, a, b, sigma, index, m, n):
    """Calculate weights using the balance heuristic method."""
    return (
        sample_counts[index]
        * calculate_normal_pdf(x, a, b, sigma, m, n, index)
        / sum(
            [
                sample_count * calculate_normal_pdf(x, a, b, sigma, m, n, i)
                for i, sample_count in enumerate(sample_counts)
            ]
        )
    )


def calculate_maximum_heuristic_weights(x, sample_counts, a, b, sigma, index, m, n):
    """Calculate weights using the maximum heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, a, b, sigma, m, n, i)
        for i, sample_count in enumerate(sample_counts)
    ]

    return float(pdf_values[index] == max(pdf_values))


def calculate_power_heuristic_weights(
    x, sample_counts, a, b, sigma, index, m, n, beta=2
):
    """Calculate weights using the power heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, a, b, sigma, m, n, i)
        for i, sample_count in enumerate(sample_counts)
    ]

    numerator = pdf_values[index] ** beta
    denominator = sum(pdf_val**beta for pdf_val in pdf_values)

    return numerator / denominator


def calculate_cutoff_heuristic_weights(
    x, sample_counts, a, b, sigma, index, m, n, alpha=0.5
):
    """Calculate weights using the cutoff heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, a, b, sigma, m, n, i)
        for i, sample_count in enumerate(sample_counts)
    ]

    q_max = max(pdf_values)
    q_index = pdf_values[index]

    if q_index < alpha * q_max:
        return 0
    else:
        denominator = sum(q_k for q_k in pdf_values if q_k >= alpha * q_max)
        return q_index / denominator


def calculate_sbert_heuristic_weights(x, sample_counts, a, b, sigma, index, m, n):
    """Calculate weights using the SBERT method."""
    return calculate_normal_pdf(x, a, b, sigma, m, n, index) / sum(
        [
            calculate_normal_pdf(x, a, b, sigma, m, n, i)
            for i, sample_count in enumerate(sample_counts)
        ]
    )


def calculate_mis_estimate(total_samples, a, b, m, n, heuristic="balance"):
    """Calculate the MIS estimate."""
    start_time = time.time()
    num_distributions = m

    samples_per_distribution = [total_samples / num_distributions] * num_distributions
    for i in range(num_distributions):
        samples_per_distribution[i] = int(np.ceil(samples_per_distribution[i]))
    total_samples = sum(samples_per_distribution)

    estimate = 0
    variance = 0
    iteration = 1
    samples_x = []
    sigma = np.array([[calculate_sigma(a[i], j) for j in range(n)] for i in range(m)])
    t = 0
    for i in range(num_distributions):
        for j in range(samples_per_distribution[i]):
            x_sample = np.array(
                [
                    np.random.normal(b[i][j_iterator], sigma[i][j_iterator])
                    for j_iterator in range(n)
                ]
            )
            y_sample = calculate_function_values(x_sample, a, b, m, n)
            weight = globals()[f"calculate_{heuristic}_heuristic_weights"](
                x_sample, samples_per_distribution, a, b, sigma, i, m, n
            )

            samples_x.append(x_sample)

            weighted_sample = (
                float(
                    weight
                    * (y_sample / calculate_normal_pdf(x_sample, a, b, sigma, m, n, i))
                )
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
    advanced_variance = sigma_variance / total_samples

    end_time = time.time()
    return estimate, samples_x, variance, advanced_variance, end_time - start_time


def plotting(sampled_points_x, a, b, m, n):
    sigma = np.array([[calculate_sigma(a[i], j) for j in range(n)] for i in range(m)])
    pdf_values = np.zeros((m, len(sampled_points_x)))
    for i in range(m):
        pdf_values[i] = [
            calculate_normal_pdf(sampled_points_x[iterator], a, b, sigma, m, n, i)
            for iterator in range(len(sampled_points_x))
        ]

    function_values = [
        calculate_function_values(sampled_points_x[iterator], a, b, m, n)
        for iterator in range(len(sampled_points_x))
    ]

    if n == 1:
        sampled_points_x = [
            float(sampled_points_x[iterator])
            for iterator in range(len(sampled_points_x))
        ]

        # Plot the sampled points
        plt.scatter(
            sampled_points_x, function_values, s=5, c="k", marker="x", label="Sech"
        )

        # Plot the PDFs
        colors = ["b", "g", "r", "c", "m", "y"]
        markers = ["o", "^", "s", "*", "+", "D"]
        for i in range(m):
            plt.scatter(
                sampled_points_x,
                pdf_values[i],
                s=5,
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                label=f"PDF {i}",
            )

        plt.legend(loc="upper right")
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        for i in range(m):
            for j in range(n):
                sampled_points_x[i][j] = float(sampled_points_x[i][j])

        # Plot the sampled points
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            [sampled_points_x[i][0] for i in range(len(sampled_points_x))],
            [sampled_points_x[i][1] for i in range(len(sampled_points_x))],
            function_values,
            s=5,
            c="k",
            marker="x",
            label="Sech",
        )

        # Plot the PDFs
        colors = ["b", "g", "r", "c", "m", "y"]
        markers = ["o", "^", "s", "*", "+", "D"]
        for i in range(m):
            ax.scatter(
                [sampled_points_x[j][0] for j in range(len(sampled_points_x))],
                [sampled_points_x[j][1] for j in range(len(sampled_points_x))],
                pdf_values[i],
                s=5,
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                label=f"PDF {i}",
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc="upper right")

    plt.title("MIS General Sum Sech")
    plt.show()


def run_mis_estimate():
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 5000

    m = 3
    n = 2
    a = np.array([[2, 2], [2, 2], [2, 2]])
    b = np.array(np.random.randint(-100, 100, size=(m, n)))

    print(f"a: {a}")
    print(f"b: {b}")

    (
        mis_estimate,
        sampled_points_x,
        variance,
        advanced_variance,
        end_time,
    ) = calculate_mis_estimate(NUM_SAMPLES, a, b, m, n)

    exact_integral = calculate_exact_integral(a, m, n)

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Advanced variance of the integral with MIS: {advanced_variance}")
    print(f"Standard deviation of the integral with MIS: {np.sqrt(variance)}")
    print(
        f"Advanced standard deviation of the integral with MIS: {np.sqrt(advanced_variance)}"
    )
    print(f"Exact result of the integral: {exact_integral}")
    print(f"Error: {exact_integral - mis_estimate}")
    print(f"Time taken: {end_time} seconds")
    plotting(sampled_points_x, a, b, m, n)


def run_mis_analysis():
    """Run the MIS analysis."""
    NUM_SAMPLES = [50, 100, 500, 1000, 5000]
    NUM_TESTS = 10
    NUM_RUNS = 100
    heuristics = ["balance", "power", "maximum", "cutoff", "sbert"]

    general_results = {"Test {}".format(test + 1): {} for test in range(NUM_TESTS)}

    for test in range(NUM_TESTS):
        m = np.random.randint(2, 10)
        n = test + 1
        a = np.array(np.random.uniform(1.8, 2, size=(m, n)))
        b = np.array(np.random.uniform(-100, 100, size=(m, n)))

        results = {
            heuristic: {num_samples: [] for num_samples in NUM_SAMPLES}
            for heuristic in heuristics
        }

        exact_integral = calculate_exact_integral(a, m, n)

        for heuristic in heuristics:
            for num_samples in NUM_SAMPLES:
                print(f"h: {heuristic}, s: {num_samples}")
                for run in range(NUM_RUNS):
                    (
                        mis_estimate,
                        _,
                        variance,
                        advanced_variance,
                        end_time,
                    ) = calculate_mis_estimate(num_samples, a, b, m, n, heuristic)

                    if run == 0:
                        mis_estimates = [mis_estimate]
                        variances = [variance]
                        advanced_variances = [advanced_variance]
                        standard_deviations = [np.sqrt(variance)]
                        advanced_standard_deviations = [np.sqrt(advanced_variance)]
                        times = [end_time]
                    else:
                        mis_estimates.append(mis_estimate)
                        variances.append(variance)
                        advanced_variances.append(advanced_variance)
                        standard_deviations.append(np.sqrt(variance))
                        advanced_standard_deviations.append(np.sqrt(advanced_variance))
                        times.append(end_time)

                results[heuristic][num_samples] = {
                    "mean of mis estimates": np.mean(mis_estimates),
                    "mean of variances": np.mean(variances),
                    "mean of advanced variances": np.mean(advanced_variances),
                    "mean of standard deviations": np.mean(standard_deviations),
                    "mean of advanced standard deviations": np.mean(
                        advanced_standard_deviations
                    ),
                    "mean of errors": np.mean(
                        [
                            exact_integral - mis_estimate
                            for mis_estimate in mis_estimates
                        ]
                    ),
                    "exact integral": exact_integral,
                    "mean of times": np.mean(times),
                }

        general_results["Test {}".format(test + 1)] = {
            "results": results,
            "a": a.tolist(),
            "b": b.tolist(),
            "m": m,
            "n": n,
        }

    open("results_mis_2_sech.txt", "w").close()

    with open("results_mis_2_sech.txt", "w") as f:
        json.dump(general_results, f, indent=4)


if __name__ == "__main__":
    run_mis_estimate()
    # run_mis_analysis()
