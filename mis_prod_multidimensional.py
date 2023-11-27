import numpy as np
import matplotlib.pyplot as plt
import time
import pdb
from mpl_toolkits.mplot3d import Axes3D


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


# Adjusted function to calculate divisor with a scaling factor
def calculate_divisor(a, b):
    average_length = np.mean(np.abs(b - a))
    scale_factor = 0.5  # Scaling factor to adjust the divisor based on hypercube size
    divisor = 2 / (1 + np.exp(-scale_factor * average_length)) - 1
    return divisor


# Function to calculate sigma, with a limit to prevent overly large spread for small hypercubes
def calculate_sigma(a, b):
    k = calculate_divisor(a, b)
    sigma_limit = 0.25 * np.abs(b - a)  # Limit sigma to 25% of hypercube size
    sigma = np.minimum([(b[i] - a[i]) / k for i in range(len(a))], sigma_limit)
    return sigma


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
            weight = calculate_balance_heuristic_weights(
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


def plot_functions_and_pdfs(sampled_points_x, a_f, b_f, c_f, a_g, b_g, c_g):
    mu_f = calculate_mu(a_f, b_f)
    mu_g = calculate_mu(a_g, b_g)
    sigma_f = calculate_sigma(a_f, b_f)
    sigma_g = calculate_sigma(a_g, b_g)

    # Calculate values for f, g, and their product
    f_values = np.array(
        [
            [
                calculate_f_x(
                    [sampled_points_x[i][0], sampled_points_x[i][1]], c_f, a_f, b_f
                )
                for j in range(100)
            ]
            for i in range(len(sampled_points_x))
        ]
    )
    g_values = np.array(
        [
            [
                calculate_g_x(
                    [sampled_points_x[i][0], sampled_points_x[i][1]], c_g, a_g, b_g
                )
                for j in range(100)
            ]
            for i in range(len(sampled_points_x))
        ]
    )
    fg_values = f_values * g_values

    # Calculate the normal PDF values for f and g
    pdf_values_f = np.array(
        [
            [
                calculate_normal_pdf(
                    [sampled_points_x[i][0], sampled_points_x[i][1]], mu_f, sigma_f
                )
                for j in range(100)
            ]
            for i in range(len(sampled_points_x))
        ]
    )
    pdf_values_g = np.array(
        [
            [
                calculate_normal_pdf(
                    [sampled_points_x[i][0], sampled_points_x[i][1]], mu_g, sigma_g
                )
                for j in range(100)
            ]
            for i in range(len(sampled_points_x))
        ]
    )

    # Plotting in 3D
    fig = plt.figure(figsize=(24, 6))

    # Plot f(x, y)
    ax1 = fig.add_subplot(1, 5, 1, projection="3d")
    ax1.scatter(
        [sampled_points_x[i][0] for i in range(len(sampled_points_x))],
        [sampled_points_x[i][1] for i in range(len(sampled_points_x))],
        [f_values[i][0] for i in range(len(sampled_points_x))],
    )
    ax1.set_title("Function f(x, y)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x, y)")

    # Plot g(x, y)
    ax2 = fig.add_subplot(1, 5, 2, projection="3d")
    ax2.scatter(
        [sampled_points_x[i][0] for i in range(len(sampled_points_x))],
        [sampled_points_x[i][1] for i in range(len(sampled_points_x))],
        [g_values[i][0] for i in range(len(sampled_points_x))],
    )
    ax2.set_title("Function g(x, y)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("g(x, y)")

    # Plot f(x, y) * g(x, y)
    ax3 = fig.add_subplot(1, 5, 3, projection="3d")
    ax3.scatter(
        [sampled_points_x[i][0] for i in range(len(sampled_points_x))],
        [sampled_points_x[i][1] for i in range(len(sampled_points_x))],
        [fg_values[i][0] for i in range(len(sampled_points_x))],
    )
    ax3.set_title("Function f(x, y) * g(x, y)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("f(x, y) * g(x, y)")

    # Plot PDF of f(x, y)
    ax4 = fig.add_subplot(1, 5, 4, projection="3d")
    ax4.scatter(
        [sampled_points_x[i][0] for i in range(len(sampled_points_x))],
        [sampled_points_x[i][1] for i in range(len(sampled_points_x))],
        [pdf_values_f[i][0] for i in range(len(sampled_points_x))],
    )
    ax4.set_title("PDF of f(x, y)")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("PDF")

    # Plot PDF of g(x, y)
    ax5 = fig.add_subplot(1, 5, 5, projection="3d")
    ax5.scatter(
        [sampled_points_x[i][0] for i in range(len(sampled_points_x))],
        [sampled_points_x[i][1] for i in range(len(sampled_points_x))],
        [pdf_values_g[i][0] for i in range(len(sampled_points_x))],
    )
    ax5.set_title("PDF of g(x, y)")
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_zlabel("PDF")

    plt.show()


def run_mis_estimate():
    """Main function to compute MIS estimate and plot the results."""
    NUM_SAMPLES = 5000

    n = 2
    a_f = np.array(np.random.uniform(-10, 0, n))
    b_f = np.array(np.random.uniform(0, 10, n))
    c_f = 0.5
    a_g = np.array(np.random.uniform(-1, 0, n))
    b_g = np.array(np.random.uniform(0, 1, n))
    c_g = 5

    (
        mis_estimate,
        sampled_points_x,
        variance,
        advanced_variance,
        end_time,
    ) = calculate_mis_estimate(
        NUM_SAMPLES, a_f, b_f, c_f, a_g, b_g, c_g, n, heuristic="balance"
    )

    exact_integral = calculate_exact_integral(a_f, b_f, c_f, a_g, b_g, c_g, n)

    print(f"Result of the integral with MIS: {mis_estimate}")
    print(f"Variance of the integral with MIS: {variance}")
    print(f"Advanced variance of the integral with MIS: {advanced_variance}")
    print(f"Standard deviation of the integral with MIS: {np.sqrt(variance)}")
    print(
        f"Advanced standard deviation of the integral with MIS: {np.sqrt(advanced_variance)}"
    )
    print(f"Error: {exact_integral - mis_estimate}")
    print(f"Exact result of the integral: {exact_integral}")
    print(f"Time taken: {end_time} seconds")

    plot_functions_and_pdfs(sampled_points_x, a_f, b_f, c_f, a_g, b_g, c_g)


if __name__ == "__main__":
    run_mis_estimate()
