import numpy as np
import mpmath
import matplotlib.pyplot as plt

def calculate_function_values(x, a, b, m, n):
    """Calculate the values of the function to be integrated."""
    total_sum = 1

    for i in range(m):
        products = [mpmath.sech(a[i][j] * (x[j] - b[i][j])) for j in range(n)]
        total_sum *= np.prod(products)

    return float(total_sum)

def normal_pdf(x, mu=0, sigma=1):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def calculate_sigma(a, index):
    sigma = np.log(2 + np.sqrt(3)) / (a[index] * np.sqrt(2 * np.log(2)))
    return sigma

def calculate_pdf_i(x, a, b, m, n, index):
    """Calculate the probability density function (PDF) using the normal distribution."""
    sigma = calculate_sigma(a, index)
    products = [normal_pdf(x[j], mu=b[index][j], sigma=sigma[j]) for j in range(n)]
    return np.prod(products)

def calculate_exact_integral(a, m, n):
    """Calculate the exact integral of the function."""
    total_sum = 1

    for i in range(m):
        products = [a[i][j] for j in range(n)]
        total_sum *= (np.pi ** n) / np.prod(products)

    return total_sum

def calculate_balance_heuristic_weights(x, sample_counts, a, b, index, m, n):
    """Calculate weights using the balance heuristic method."""
    return (
        sample_counts[index]
        * calculate_pdf_i(x, a, b, m, n, index)
        / sum([
            sample_count * calculate_pdf_i(x, a, b, m, n, i)
            for i, sample_count in enumerate(sample_counts)
        ])
    )

def calculate_maximum_heuristic_weights(x, sample_counts, a, b, index, m, n):
    """Calculate weights using the maximum heuristic method."""
    pdf_values = [
        sample_count * calculate_pdf_i(x, a, b, m, n, i)
        for i, sample_count in enumerate(sample_counts)
    ]

    return float(pdf_values[index] == max(pdf_values))

def calculate_power_heuristic_weights(x, sample_counts, a, b, index, m, n, beta=2):
    """Calculate weights using the power heuristic method."""
    pdf_values = [
        sample_count * calculate_pdf_i(x, a, b, m, n, i)
        for i, sample_count in enumerate(sample_counts)
    ]

    numerator = pdf_values[index] ** beta
    denominator = sum(pdf_val ** beta for pdf_val in pdf_values)

    return numerator / denominator

def calculate_cutoff_heuristic_weights(x, sample_counts, a, b, index, m, n, alpha=0.5):
    """Calculate weights using the cutoff heuristic method."""
    pdf_values = [
        sample_count * calculate_pdf_i(x, a, b, m, n, i)
        for i, sample_count in enumerate(sample_counts)
    ]

    q_max = max(pdf_values)
    q_index = pdf_values[index]

    if q_index < alpha * q_max:
        return 0
    else:
        denominator = sum(q_k for q_k in pdf_values if q_k >= alpha * q_max)
        return q_index / denominator

def calculate_sbert_heuristic_weights(x, sample_counts, a, b, index, m, n):
    """Calculate weights using the SBERT method."""
    return (
        calculate_pdf_i(x, a, b, m, n, index)
        / sum([
            calculate_pdf_i(x, a, b, m, n, i)
            for i, sample_count in enumerate(sample_counts)
        ])
    )

def calculate_mis_estimate(total_samples, a, b, m, n, heuristic='balance'):
    """Calculate the MIS estimate."""
    num_distributions = m

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
            x_sample = np.array([np.random.normal(b[i][j_iterator], a[i][j_iterator]) for j_iterator in range(n)])
            y_sample = calculate_function_values(x_sample, a, b, m, n)
            weight = globals()[f"calculate_{heuristic}_heuristic_weights"](x_sample, samples_per_distribution, a, b, i, m, n)

            sampled_points_x.append(x_sample)
            sampled_points_y.append(y_sample)

            weighted_sample = (float(weight * (y_sample / calculate_pdf_i(x_sample, a, b, m, n, i))) / samples_per_distribution[i]) * total_samples
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

def plotting(sampled_points_x, sampled_points_y, a, b, m, n):
    pdf_values_1 = [calculate_pdf_i(sampled_points_x[iterator], a, b, m, n, 0) for iterator in range(len(sampled_points_x))]
    pdf_values_2 = [calculate_pdf_i(sampled_points_x[iterator], a, b, m, n, 1) for iterator in range(len(sampled_points_x))]
    pdf_values_3 = [calculate_pdf_i(sampled_points_x[iterator], a, b, m, n, 2) for iterator in range(len(sampled_points_x))]

    sampled_points_x = [float(sampled_points_x[iterator]) for iterator in range(len(sampled_points_x))]

    # Plot the sampled points
    plt.scatter(sampled_points_x, sampled_points_y, s=1, c='k', label='Sampled Points')

    # Plot the PDFs
    plt.scatter(sampled_points_x, pdf_values_1, s=5, c='r', marker='o', label='PDF 1')
    plt.scatter(sampled_points_x, pdf_values_2, s=5, c='g', marker='^', label='PDF 2')
    plt.scatter(sampled_points_x, pdf_values_3, s=5, c='b', marker='s', label='PDF 3')

    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('MIS General 2')
    plt.show()

def run_mis_analysis():
    """Run the MIS analysis."""
    NUM_SAMPLES = [10, 25, 50, 100]
    NUM_RUNS_PER_HEURISTIC = 5000
    heuristics=["balance", "power", "maximum", "cutoff", "sbert"]

    results = {heuristic : { num_samples : [] for num_samples in NUM_SAMPLES } for heuristic in heuristics}
    errors = {heuristic : { num_samples : [] for num_samples in NUM_SAMPLES } for heuristic in heuristics}

    for heuristic in heuristics:
        for num_samples in NUM_SAMPLES:
            for run in range(NUM_RUNS_PER_HEURISTIC):
                m = np.random.randint(2, 4)
                n = np.random.randint(2, 4)
                a = np.array(np.random.uniform(0.1, 1, size=(m, n)))
                b = np.array(np.random.uniform(-1, 1, size=(m, n)))

                mis_estimate, _, _, variance, alternate_variance = calculate_mis_estimate(
                    num_samples, a, b, m, n
                )
                exact_integral = calculate_exact_integral(a, m, n)

                if run == 0:
                    mis_estimates = [mis_estimate]
                    variances = [variance]
                    alternate_variances = [alternate_variance]
                    exact_integrals = [exact_integral]
                else:
                    mis_estimates.append(mis_estimate)
                    variances.append(variance)
                    alternate_variances.append(alternate_variance)
                    exact_integrals.append(exact_integral)

            errors_per_heuristic_and_num_samples = []
            for i in range(NUM_RUNS_PER_HEURISTIC):
                errors_per_heuristic_and_num_samples.append(exact_integrals[i] - mis_estimates[i])

            results[heuristic][num_samples] = { "mean of MIS estimates" : np.mean(mis_estimates),
                                   "mean of variances" : np.mean(variances),
                                   "mean of alternate variances" : np.mean(alternate_variances),
                                   "mean of exact integrals" : np.mean(exact_integrals),
                                   "mean of errors" : np.mean(errors_per_heuristic_and_num_samples),
                                   "standard deviation of errors" : np.std(errors_per_heuristic_and_num_samples)
                                }

            errors[heuristic][num_samples] = errors_per_heuristic_and_num_samples

    open('results_prod_sech.txt', 'w').close()
    open('errors_prod_sech.txt', 'w').close()

    with open('results_prod_sech.txt', 'w') as f:
        print(results, file=f)

    with open('errors_prod_sech.txt', 'w') as f:
        print(errors, file=f)

    # for heuristic in heuristics:
    #     for num_samples in NUM_SAMPLES:
    #       plt.hist(errors[heuristic][num_samples], bins=100)
    #       plt.xlabel('Error')
    #       plt.ylabel('Frequency')
    #       plt.title(f'Error Histogram ({heuristic}, {num_samples} samples)')
    #       plt.show()


def run_mis_estimate():
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 50

    m = np.random.randint(2, 10)
    n = 1
    a = np.array(np.random.uniform(0.1, 1, size=(m, n)))
    b = np.array(np.random.uniform(-1, 1, size=(m, n)))

    print(f"a: {a}")
    print(f"b: {b}")

    mis_estimate, sampled_points_x, sampled_points_y, variance, alternate_variance = calculate_mis_estimate(
        NUM_SAMPLES, a, b, m, n
    )

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Alternate variance of the integral with MIS: {alternate_variance}")
    print(f"Exact result of the integral: {calculate_exact_integral(a, m, n)}")

if __name__ == "__main__":
    # run_mis_estimate()
    run_mis_analysis()
