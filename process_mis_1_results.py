import json
from collections import defaultdict
import numpy as np

# Read the JSON file
with open('results_mis_1.txt') as f:
    data = json.load(f)

variances, errors, std_devs, mis_estimate_differences = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))

# Iterating through the JSON data to extract the necessary statistics
for test in data.values():
    quad_result = test['test_values']['quad_result']
    for heuristic, sample_data in test.items():
        if heuristic == 'test_values':
            continue  # Skipping the 'test_values' section
        for sample_size, stats in sample_data.items():
            variances[heuristic][sample_size].append(stats['mean of variances'])
            errors[heuristic][sample_size].append(stats['mean of errors'])
            std_devs[heuristic][sample_size].append(stats['mean of standard deviations'])
            mis_estimate = stats['mean of mis estimate']
            difference = abs(mis_estimate - quad_result)
            mis_estimate_differences[heuristic][sample_size].append(difference)

# Calculating the means for each heuristic and sample size
mean_variances = {heuristic: {size: np.mean(values) for size, values in sizes.items()} for heuristic, sizes in variances.items()}
mean_errors = {heuristic: {size: np.mean(values) for size, values in sizes.items()} for heuristic, sizes in errors.items()}
mean_std_devs = {heuristic: {size: np.mean(values) for size, values in sizes.items()} for heuristic, sizes in std_devs.items()}
mean_mis_estimate_differences = {heuristic: {size: np.mean(values) for size, values in sizes.items()} for heuristic, sizes in mis_estimate_differences.items()}

# Printing the means
print("Analysis of the results per heuristic and sample size:")
print('Mean of variances:')
print(mean_variances)
print('Mean of errors:')
print(mean_errors)
print('Mean of standard deviations:')
print(mean_std_devs)
print('Mean of mis estimate differences:')
print(mean_mis_estimate_differences)

print("*" * 10)

# Iterate through tests
print("Analysis of the results per heuristic in each test:")
for i, test in enumerate(data.values()):
    variances, errors, std_devs, mis_estimate_differences = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    quad_result = test['test_values']['quad_result']
    for heuristic, sample_data in test.items():
        if heuristic == 'test_values':
            continue
        for sample_size, stats in sample_data.items():
            variances[heuristic].append(stats['mean of variances'])
            errors[heuristic].append(stats['mean of errors'])
            std_devs[heuristic].append(stats['mean of standard deviations'])
            mis_estimate = stats['mean of mis estimate']
            difference = abs(mis_estimate - quad_result)
            mis_estimate_differences[heuristic].append(difference)

    # Calculating the means for each heuristic and sample size
    mean_variances = {heuristic: np.mean(values) for heuristic, values in variances.items()}
    mean_errors = {heuristic: np.mean(values) for heuristic, values in errors.items()}
    mean_std_devs = {heuristic: np.mean(values) for heuristic, values in std_devs.items()}
    mean_mis_estimate_differences = {heuristic: np.mean(values) for heuristic, values in mis_estimate_differences.items()}
    print('Test: ', i + 1)
    print('Mean of variances:')
    print(mean_variances)
    print('Mean of errors:')
    print(mean_errors)
    print('Mean of standard deviations:')
    print(mean_std_devs)
    print('Mean of mis estimate differences:')
    print(mean_mis_estimate_differences)

print("*" * 10)

# Iterating through heuristics
print("Analysis of the results per heuristic:")
balance_variances, balance_errors, balance_std_devs, balance_mis_estimate_differences = [], [], [], []
power_variances, power_errors, power_std_devs, power_mis_estimate_differences = [], [], [], []
maximum_variances, maximum_errors, maximum_std_devs, maximum_mis_estimate_differences = [], [], [], []
cutoff_variances, cutoff_errors, cutoff_std_devs, cutoff_mis_estimate_differences = [], [], [], []
sbert_variances, sbert_errors, sbert_std_devs, sbert_mis_estimate_differences = [], [], [], []

for test in data.values():
    quad_result = test['test_values']['quad_result']
    for heuristic, sample_data in test.items():
        if heuristic == 'test_values':
            continue
        for sample_size, stats in sample_data.items():
            if heuristic == 'balance':
                balance_variances.append(stats['mean of variances'])
                balance_errors.append(stats['mean of errors'])
                balance_std_devs.append(stats['mean of standard deviations'])
                mis_estimate = stats['mean of mis estimate']
                difference = abs(mis_estimate - quad_result)
                balance_mis_estimate_differences.append(difference)
            elif heuristic == 'power':
                power_variances.append(stats['mean of variances'])
                power_errors.append(stats['mean of errors'])
                power_std_devs.append(stats['mean of standard deviations'])
                mis_estimate = stats['mean of mis estimate']
                difference = abs(mis_estimate - quad_result)
                power_mis_estimate_differences.append(difference)
            elif heuristic == 'maximum':
                maximum_variances.append(stats['mean of variances'])
                maximum_errors.append(stats['mean of errors'])
                maximum_std_devs.append(stats['mean of standard deviations'])
                mis_estimate = stats['mean of mis estimate']
                difference = abs(mis_estimate - quad_result)
                maximum_mis_estimate_differences.append(difference)
            elif heuristic == 'cutoff':
                cutoff_variances.append(stats['mean of variances'])
                cutoff_errors.append(stats['mean of errors'])
                cutoff_std_devs.append(stats['mean of standard deviations'])
                mis_estimate = stats['mean of mis estimate']
                difference = abs(mis_estimate - quad_result)
                cutoff_mis_estimate_differences.append(difference)
            elif heuristic == 'sbert':
                sbert_variances.append(stats['mean of variances'])
                sbert_errors.append(stats['mean of errors'])
                sbert_std_devs.append(stats['mean of standard deviations'])
                mis_estimate = stats['mean of mis estimate']
                difference = abs(mis_estimate - quad_result)
                sbert_mis_estimate_differences.append(difference)

# Calculating the means for each heuristic and sample size
mean_balance_variances = np.mean(balance_variances)
mean_balance_errors = np.mean(balance_errors)
mean_balance_std_devs = np.mean(balance_std_devs)
mean_balance_mis_estimate_differences = np.mean(balance_mis_estimate_differences)

mean_power_variances = np.mean(power_variances)
mean_power_errors = np.mean(power_errors)
mean_power_std_devs = np.mean(power_std_devs)
mean_power_mis_estimate_differences = np.mean(power_mis_estimate_differences)

mean_maximum_variances = np.mean(maximum_variances)
mean_maximum_errors = np.mean(maximum_errors)
mean_maximum_std_devs = np.mean(maximum_std_devs)
mean_maximum_mis_estimate_differences = np.mean(maximum_mis_estimate_differences)

mean_cutoff_variances = np.mean(cutoff_variances)
mean_cutoff_errors = np.mean(cutoff_errors)
mean_cutoff_std_devs = np.mean(cutoff_std_devs)
mean_cutoff_mis_estimate_differences = np.mean(cutoff_mis_estimate_differences)

mean_sbert_variances = np.mean(sbert_variances)
mean_sbert_errors = np.mean(sbert_errors)
mean_sbert_std_devs = np.mean(sbert_std_devs)
mean_sbert_mis_estimate_differences = np.mean(sbert_mis_estimate_differences)

print('Mean of variances:')
print('balance: ', mean_balance_variances)
print('power: ', mean_power_variances)
print('maximum: ', mean_maximum_variances)
print('cutoff: ', mean_cutoff_variances)
print('sbert: ', mean_sbert_variances)

print('Mean of errors:')
print('balance: ', mean_balance_errors)
print('power: ', mean_power_errors)
print('maximum: ', mean_maximum_errors)
print('cutoff: ', mean_cutoff_errors)
print('sbert: ', mean_sbert_errors)

print('Mean of standard deviations:')
print('balance: ', mean_balance_std_devs)
print('power: ', mean_power_std_devs)
print('maximum: ', mean_maximum_std_devs)
print('cutoff: ', mean_cutoff_std_devs)
print('sbert: ', mean_sbert_std_devs)

print('Mean of mis estimate differences:')
print('balance: ', mean_balance_mis_estimate_differences)
print('power: ', mean_power_mis_estimate_differences)
print('maximum: ', mean_maximum_mis_estimate_differences)
print('cutoff: ', mean_cutoff_mis_estimate_differences)
print('sbert: ', mean_sbert_mis_estimate_differences)
