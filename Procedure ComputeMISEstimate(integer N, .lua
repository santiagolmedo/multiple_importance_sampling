Procedure ComputeMISEstimate(integer N, array alfai, array mu, array sigma, array a, array b)
Parameters of input: 
    N, size of the sample
    alfai, array of alfa values
    mu, array of mu values
    sigma, array of sigma values
    a, array of 'a' values
    b, array of 'b' values
Parameters of output:
    F, MIS estimate value
    sampled_points_X, array of X values
    sampled_points_Y, array of Y values
    variance, variance of the estimate

1. K = length of alfai
2. ni = round(alfai * N)
3. F = 0
5. variance = 0

6. For i = 1 to K do
    6.1 total_sum = 0
    6.2 For j = 1 to ni[i] do
        6.2.1 X = sigma[i] * randomNormalValue + mu[i]
        6.2.2 Y = compute_function_values(X, a, b)
        6.2.3 weights = compute_balance_heuristic_weights(X, ni, mu, sigma)
        6.2.6 value_ij = weights[i] * (Y / compute_p_k(X, mu[i], sigma[i]))
        6.2.7 total_sum = total_sum + value_ij
        6.2.8 variance = variance + value_ij^2
    6.3 F = F + total_sum / ni[i]

7. variance = variance / (N * (N - 1)) - F^2 / (N - 1)

8. Return F, sampled_points_X, sampled_points_Y, variance
End Procedure