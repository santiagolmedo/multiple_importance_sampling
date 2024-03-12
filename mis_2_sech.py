import numpy as np
import matplotlib.pyplot as plt
import time
import json
import pdb


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
    return estimate, samples_x, variance, end_time - start_time


def calculate_mis_estimate_sbert(total_samples, a, b, m, n):
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

    # Stops for variance calculation
    first_stop = int(total_samples * 0.2)
    # after first stop, stop every 0.1 of the remaining samples
    stop_interval = int((total_samples - first_stop) // 10)
    stops = [first_stop] + [first_stop + stop_interval * i for i in range(1, 10)]

    ni_per_distribution = {
        stop: [0 for i in range(num_distributions)] for stop in range(len(stops))
    }
    ni_per_distribution[0] = [
        int(first_stop / num_distributions) for i in range(num_distributions)
    ]
    stop_counter = 0
    estimation_per_distribution = np.zeros(num_distributions)
    variance_per_distribution = np.zeros(num_distributions)
    iteration_per_distribution = np.zeros(num_distributions)
    std_dev_per_distribution = np.zeros(num_distributions)

    for stop in range(len(stops)):
        for i in range(num_distributions):
            for ni_iterator in range(ni_per_distribution[stop_counter][i]):
                x_sample = np.array(
                    [
                        np.random.normal(b[i][j_iterator], sigma[i][j_iterator])
                        for j_iterator in range(n)
                    ]
                )
                y_sample = calculate_function_values(x_sample, a, b, m, n)
                weight = calculate_sbert_heuristic_weights(
                    x_sample, samples_per_distribution, a, b, sigma, i, m, n
                )

                samples_x.append(x_sample)

                weighted_sample = (
                    float(
                        weight
                        * (
                            y_sample
                            / calculate_normal_pdf(x_sample, a, b, sigma, m, n, i)
                        )
                    )
                    / samples_per_distribution[i]
                ) * total_samples

                estimate += weighted_sample
                variance += weighted_sample**2
                estimation_per_distribution[i] += weighted_sample
                variance_per_distribution[i] += weighted_sample**2
                iteration_per_distribution[i] += 1
                iteration += 1

                # skip if it's the last stop
                if stop == len(stops) - 1:
                    continue
                if ni_iterator == (ni_per_distribution[stop_counter][i] - 1):
                    partial_estimate = (
                        estimation_per_distribution[i] / iteration_per_distribution[i]
                    )
                    partial_variance = (
                        variance_per_distribution[i]
                        / (
                            iteration_per_distribution[i] ** 2
                            - iteration_per_distribution[i]
                        )
                    ) - (partial_estimate**2 / (iteration_per_distribution[i] - 1))
                    partial_std_dev = np.sqrt(partial_variance)
                    std_dev_per_distribution[i] += partial_std_dev
                if iteration - 1 == stops[stop_counter]:
                    if stop_counter not in ni_per_distribution:
                        print("stop_counter not in ni_per_distribution")
                        break
                    total_std_dev = sum(std_dev_per_distribution)
                    stop_counter += 1
                    for distribution in range(num_distributions):
                        contributed_std_dev = std_dev_per_distribution[distribution]
                        if contributed_std_dev > 0:
                            ni_per_distribution[stop_counter][distribution] = int(
                                (contributed_std_dev / total_std_dev) * stop_interval
                            )
                        else:
                            ni_per_distribution[stop_counter][distribution] = 0
                    # If any distribution has 0 samples, distribute the samples equally
                    for distribution in range(num_distributions):
                        if ni_per_distribution[stop_counter][distribution] == 0:
                            ni_per_distribution[stop_counter] = [
                                int(stop_interval / num_distributions)
                                for i in range(num_distributions)
                            ]
                            break

    estimate = estimate / total_samples
    variance = (variance / (total_samples**2 - total_samples)) - (
        estimate**2 / (total_samples - 1)
    )

    print(f"std_dev_per_distribution: {std_dev_per_distribution}")
    print(f"ni_per_distribution: {ni_per_distribution}")
    ni_per_distribution_sum = {
        key: sum(ni_per_distribution[key]) for key in ni_per_distribution
    }
    print(f"ni_per_distribution_sum: {ni_per_distribution_sum}")

    end_time = time.time()
    return estimate, samples_x, variance, end_time - start_time


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
    NUM_SAMPLES = 50000

    m = 4
    n = 4
    a = np.array(np.random.uniform(1.8, 2, size=(m, n)))
    b = np.array(np.random.uniform(-100, 100, size=(m, n)))

    print(f"a: {a}")
    print(f"b: {b}")

    exact_integral = calculate_exact_integral(a, m, n)

    (
        mis_estimate,
        sampled_points_x,
        variance,
        end_time,
    ) = calculate_mis_estimate_sbert(NUM_SAMPLES, a, b, m, n)

    print(f"Sbert Results")
    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Standard deviation of the integral with MIS: {np.sqrt(variance)}")
    print(f"Exact result of the integral: {exact_integral}")
    print(f"Error: {exact_integral - mis_estimate}")
    print(f"Time taken: {end_time} seconds")
    print("*" * 50)

    (
        mis_estimate,
        sampled_points_x,
        variance,
        end_time,
    ) = calculate_mis_estimate(NUM_SAMPLES, a, b, m, n, "balance")
    print(f"Balance Results")
    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Standard deviation of the integral with MIS: {np.sqrt(variance)}")
    print(f"Exact result of the integral: {exact_integral}")
    print(f"Error: {exact_integral - mis_estimate}")
    print(f"Time taken: {end_time} seconds")


def run_mis_analysis():
    """Run the MIS analysis."""
    NUM_SAMPLES = [500, 1000, 5000, 10000, 50000]
    NUM_TESTS = 5
    NUM_RUNS = 25
    heuristics = ["balance", "power", "maximum", "cutoff", "sbert"]

    general_results = {"Test {}".format(test + 1): {} for test in range(NUM_TESTS)}

    for test in range(NUM_TESTS):
        print("*" * 10)
        m = np.random.randint(3, 10)
        n = (test + 1) * 2
        print(f"Test {test + 1}")
        print(f"m: {m}, n: {n}")
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
                    if heuristic == "sbert":
                        (
                            mis_estimate,
                            _,
                            variance,
                            end_time,
                        ) = calculate_mis_estimate_sbert(num_samples, a, b, m, n)
                    else:
                        (
                            mis_estimate,
                            _,
                            variance,
                            end_time,
                        ) = calculate_mis_estimate(num_samples, a, b, m, n, heuristic)

                    if run == 0:
                        mis_estimates = [mis_estimate]
                        variances = [variance]
                        standard_deviations = [np.sqrt(variance)]
                        times = [end_time]
                    else:
                        mis_estimates.append(mis_estimate)
                        variances.append(variance)
                        standard_deviations.append(np.sqrt(variance))
                        times.append(end_time)

                results[heuristic][num_samples] = {
                    "mean of mis estimates": np.mean(mis_estimates),
                    "mean of variances": np.mean(variances),
                    "mean of standard deviations": np.mean(standard_deviations),
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

    with open("results_mis_2_sech.txt", "w") as f:
        json.dump(general_results, f, indent=4)


if __name__ == "__main__":
    # run_mis_estimate()
    run_mis_analysis()
