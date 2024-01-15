#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import struct
from scipy.signal import resample
from joblib import Parallel, delayed

def load_and_clean_data(entries_file, traces_file, num_entries=300, resample_points=5000):
    avg_entries = np.load(entries_file)
    subset_traces = np.load(traces_file)

    selected_avg_entries = avg_entries[:num_entries]
    selected_subset_traces = subset_traces[:num_entries]

    resampled_subset_traces = resample(selected_subset_traces, resample_points, axis=1)

    # Remove NaN values
    nan_indices = np.argwhere(np.isnan(selected_avg_entries))
    cleaned_avg_entries = np.delete(selected_avg_entries, nan_indices, axis=0)
    cleaned_subset_traces = np.delete(resampled_subset_traces, nan_indices, axis=0)

    return cleaned_avg_entries, cleaned_subset_traces

def float_to_binary(num):
    return ''.join(f'{c:08b}' for c in struct.pack('!f', num))

def hamming_weight(binary_str):
    return binary_str.count('1')

def calculate_correlations(traces, inputs, hypothetical_weights):
    num_sample_points = traces.shape[1]
    hws = np.array([hamming_weight(float_to_binary(input_value * weight)) for input_value in inputs for weight in hypothetical_weights])
    hws = hws.reshape(len(inputs), len(hypothetical_weights))

    correlations = np.array([[pearsonr(traces[:, sample_point], hws[:, weight_idx])[0] for sample_point in range(num_sample_points)] for weight_idx in range(len(hypothetical_weights))])
    correlations = np.nan_to_num(correlations)

    return np.max(np.abs(correlations), axis=1)

def plot_correlations(hypothetical_weights, max_correlations):
    plt.figure(figsize=(12, 6))
    plt.plot(hypothetical_weights, max_correlations, marker='o')
    plt.title('Maximum Correlation of Hamming Weight with Each Trace Point')
    plt.xlabel('Hypothetical Weight')
    plt.ylabel('Maximum Correlation Coefficient')
    plt.grid()
    plt.show()

def main(entries_file, traces_file):
    hypothetical_weights = np.arange(0.7, 0.9, 0.001)
    cleaned_avg_entries, cleaned_subset_traces = load_and_clean_data(entries_file, traces_file)

    # Parallel processing
    num_cores = 4  # Adjust based on your HPC environment
    max_correlations = Parallel(n_jobs=num_cores)(delayed(calculate_correlations)(cleaned_subset_traces, cleaned_avg_entries, [weight]) for weight in hypothetical_weights)

    plot_correlations(hypothetical_weights, np.concatenate(max_correlations))

if __name__ == "__main__":
    main("avg_entries.npy", "avg_traces.npy")

