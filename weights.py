def compute_balance_heuristic_weights(X, ni, mu, sigma):
    """Compute the weights using the balance heuristic."""
    K = len(mu)
    pi_X_values = [compute_p_k(X, mu[k], sigma[k]) for k in range(K)]
    return [
        (ni[i] * pi_X_values[i]) / sum([(ni[k] * pi_X_values[k]) for k in range(K)])
        for i in range(K)
    ]


def compute_power_heuristic_weights(X, ni, mu, sigma, beta=2):
    """Compute the weights using the power heuristic."""
    K = len(mu)
    pi_X_values = [compute_p_k(X, mu[k], sigma[k]) for k in range(K)]
    return [
        (ni[i] * pi_X_values[i]) ** beta
        / sum([(ni[k] * pi_X_values[k]) ** beta for k in range(K)])
        for i in range(K)
    ]


def compute_cutoff_heuristic_weights(X, ni, mu, sigma):
    """Compute the weights using the cutoff heuristic."""
    K = len(mu)
    pi_X_values = [compute_p_k(X, mu[k], sigma[k]) for k in range(K)]
    max_val_index = np.argmax([ni[k] * pi_X_values[k] for k in range(K)])
    return [1 if i == max_val_index else 0 for i in range(K)]


def compute_maximum_heuristic_weights(X, ni, mu, sigma):
    """Compute the weights using the maximum heuristic."""
    K = len(mu)
    pi_X_values = [compute_p_k(X, mu[k], sigma[k]) for k in range(K)]
    max_val = max([ni[k] * pi_X_values[k] for k in range(K)])
    return [(ni[i] * pi_X_values[i]) / max_val for i in range(K)]