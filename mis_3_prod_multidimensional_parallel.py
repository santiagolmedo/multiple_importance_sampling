import numpy as np
import time
import json
import concurrent.futures


def calculate_f_x(x, c_f, a_f, b_f):
    # Hypercube F in n dimensions
    for i in range(len(x)):
        if not (a_f[i] <= x[i] <= b_f[i]):
            return 0
    return c_f


def calculate_g_x(x, c_g, a_g, b_g):
    # Hypercube G in n dimensions
    for i in range(len(x)):
        if not (a_g[i] <= x[i] <= b_g[i]):
            return 0
    return c_g


def calculate_exact_integral(a_f, b_f, c_f, a_g, b_g, c_g, n):
    prod = [max(0, min(b_f[i], b_g[i]) - max(a_f[i], a_g[i])) for i in range(n)]
    return c_f * c_g * np.prod(prod)


def normal_pdf(x, mu=0, sigma=1):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def calculate_normal_pdf(x, mu, sigma):
    products = [normal_pdf(x[j], mu[j], sigma[j]) for j in range(len(x))]
    return np.prod(products)


def calculate_mu(a, b):
    return [(a[i] + b[i]) / 2 for i in range(len(a))]


def calculate_sigma(a, b):
    return [(b[i] - a[i]) / 4 for i in range(len(a))]


def calculate_balance_heuristic_weights(x, sample_counts, mu, sigma, index):
    """Calculate weights using the balance heuristic method."""
    return (
        sample_counts[index]
        * calculate_normal_pdf(x, mu, sigma)
        / sum(
            [
                sample_count * calculate_normal_pdf(x, mu, sigma)
                for i, sample_count in enumerate(sample_counts)
            ]
        )
    )


def calculate_maximum_heuristic_weights(x, sample_counts, mu, sigma, index):
    """Calculate weights using the maximum heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, mu, sigma)
        for i, sample_count in enumerate(sample_counts)
    ]

    return float(pdf_values[index] == max(pdf_values))


def calculate_power_heuristic_weights(x, sample_counts, mu, sigma, index, beta=2):
    """Calculate weights using the power heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, mu, sigma)
        for i, sample_count in enumerate(sample_counts)
    ]

    numerator = pdf_values[index] ** beta
    denominator = sum(pdf_val**beta for pdf_val in pdf_values)

    return numerator / denominator


def calculate_cutoff_heuristic_weights(x, sample_counts, mu, sigma, index, alpha=0.5):
    """Calculate weights using the cutoff heuristic method."""
    pdf_values = [
        sample_count * calculate_normal_pdf(x, mu, sigma)
        for i, sample_count in enumerate(sample_counts)
    ]

    q_max = max(pdf_values)
    q_index = pdf_values[index]

    if q_index < alpha * q_max:
        return 0
    else:
        denominator = sum(q_k for q_k in pdf_values if q_k >= alpha * q_max)
        return q_index / denominator


def calculate_sbert_heuristic_weights(x, sample_counts, mu, sigma, index):
    """Calculate weights using the SBERT method."""
    return calculate_normal_pdf(x, mu, sigma) / sum(
        [
            calculate_normal_pdf(x, mu, sigma)
            for i, sample_count in enumerate(sample_counts)
        ]
    )


def calculate_mis_estimate(
    total_samples, a_f, b_f, c_f, a_g, b_g, c_g, n, heuristic="balance"
):
    """Calculate the MIS estimate."""
    start_time = time.time()
    num_distributions = 2

    samples_per_distribution = [total_samples / num_distributions] * num_distributions
    for i in range(num_distributions):
        samples_per_distribution[i] = int(np.ceil(samples_per_distribution[i]))
    total_samples = sum(samples_per_distribution)

    estimate = 0
    variance = 0
    iteration = 1
    samples_x = []
    mu_f = calculate_mu(a_f, b_f)
    mu_g = calculate_mu(a_g, b_g)
    sigma_f = calculate_sigma(a_f, b_f)
    sigma_g = calculate_sigma(a_g, b_g)
    t = 0
    for i in range(num_distributions):
        mu = mu_f if i == 0 else mu_g
        sigma = sigma_f if i == 0 else sigma_g

        for j in range(samples_per_distribution[i]):
            x_sample = np.random.normal(mu, sigma)
            y_sample = calculate_f_x(x_sample, c_f, a_f, b_f) * calculate_g_x(
                x_sample, c_g, a_g, b_g
            )
            weight = globals()[f"calculate_{heuristic}_heuristic_weights"](
                x_sample, samples_per_distribution, mu, sigma, i
            )

            samples_x.append(x_sample)

            weighted_sample = (
                float(weight * (y_sample / calculate_normal_pdf(x_sample, mu, sigma)))
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


def run_single_test(test, NUM_DIMENSIONS, NUM_SAMPLES, NUM_RUNS, heuristics):
    n = NUM_DIMENSIONS[test]
    a_f = np.array(np.random.uniform(-10, 0, n))
    b_f = np.array(np.random.uniform(0, 10, n))
    c_f = 0.5
    a_g = np.array(np.random.uniform(-0.5, 0, n))
    b_g = np.array(np.random.uniform(0, 0.5, n))
    c_g = 5

    results = {
        heuristic: {num_samples: [] for num_samples in NUM_SAMPLES}
        for heuristic in heuristics
    }

    exact_integral = calculate_exact_integral(a_f, b_f, c_f, a_g, b_g, c_g, n)

    for heuristic in heuristics:
        for num_samples in NUM_SAMPLES:
            print(f"h: {heuristic}, s: {num_samples}")
            for run in range(NUM_RUNS):
                (
                    mis_estimate,
                    sampled_points_x,
                    variance,
                    advanced_variance,
                    end_time,
                ) = calculate_mis_estimate(
                    num_samples,
                    a_f,
                    b_f,
                    c_f,
                    a_g,
                    b_g,
                    c_g,
                    n,
                    heuristic=heuristic,
                )

                if run == 0:
                    mis_estimates = [mis_estimate]
                    variances = [variance]
                    advanced_variances = [advanced_variance]
                    times = [end_time]
                else:
                    mis_estimates.append(mis_estimate)
                    variances.append(variance)
                    advanced_variances.append(advanced_variance)
                    times.append(end_time)

            results[heuristic][num_samples] = {
                "mean of mis estimate": np.mean(mis_estimates),
                "mean of variance": np.mean(variances),
                "mean of advanced variance": np.mean(advanced_variances),
                "mean of standard deviation": np.mean(np.sqrt(variances)),
                "mean of advanced standard deviation": np.mean(
                    np.sqrt(advanced_variances)
                ),
                "mean of error": np.mean(
                    [
                        exact_integral - mis_estimate
                        for mis_estimate in mis_estimates
                    ]
                ),
                "mean of time taken": np.mean(times),
                "exact integral": exact_integral,
            }

    return test, results


def run_mis_analysis():
    NUM_DIMENSIONS = [2, 3, 4, 5, 6, 10, 12, 15, 18, 20]
    NUM_SAMPLES = [25, 50, 100, 500, 1000, 5000, 50000]
    NUM_TESTS = 10
    NUM_RUNS = 100
    heuristics = ["balance", "power", "maximum", "cutoff", "sbert"]

    general_results = {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_test, test, NUM_DIMENSIONS, NUM_SAMPLES, NUM_RUNS, heuristics) for test in range(NUM_TESTS)]
        for future in concurrent.futures.as_completed(futures):
            test, test_results = future.result()
            general_results["Test {}".format(test + 1)] = test_results

    with open('results_mis_3_prod_multidimensional_aux.txt', 'w') as file:
        json.dump(general_results, file)

if __name__ == "__main__":
    run_mis_analysis()
