import numpy as np
import mpmath
import matplotlib.pyplot as plt
import pdb

def calculate_function_values(x, a, b, m, n):
    """Calculate the values of the function to be integrated."""
    total_sum = 0

    for i in range(m):
        products = [mpmath.sech(a[i][j] * (x[j] - b[i][j])) for j in range(n)]
        total_sum += np.prod(products)

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
    total_sum = 0

    for i in range(m):
        products = [a[i][j] for j in range(n)]
        total_sum += (np.pi ** n) / np.prod(products)

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

# def calculate_maximum_heuristic_weights(x, sample_counts, a, b, index, m, n):
#     """Calculate weights using the maximum heuristic method."""
#     pdf_values = [
#         sample_count * calculate_pdf_i(x, a, b, m, n, i)
#         for i, sample_count in enumerate(sample_counts)
#     ]

#     return float(pdf_values[index] == max(pdf_values))

def calculate_mis_estimate(total_samples, a, b, m, n):
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
            weight = calculate_balance_heuristic_weights(x_sample, samples_per_distribution, a, b, i, m, n)

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


def main():
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 50

    a = np.array([[1], [1], [1]])
    b = np.array([[0], [0], [0]])
    m = len(a)
    n = len(a[0])
    print(f"a: {a}")
    print(f"b: {b}")

    mis_estimate, sampled_points_x, sampled_points_y, variance, alternate_variance = calculate_mis_estimate(
        NUM_SAMPLES, a, b, m, n
    )

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Alternate variance of the integral with MIS: {alternate_variance}")
    print(f"Exact result of the integral: {calculate_exact_integral(a, m, n)}")
    plotting(sampled_points_x, sampled_points_y, a, b, m, n)

if __name__ == "__main__":
    main()
