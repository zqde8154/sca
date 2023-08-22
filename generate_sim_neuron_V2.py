"""
Script part of SCANN-ex github project
Related to "Practical Introduction to Side-Channel Extraction of Deep Neural Network Parameters" publication
"""

import sys, os, array, time, math
import numpy as np
import matplotlib.pyplot as plt
import IPython
import pickle
import random

from Tools import *
from Variable_constructor import *


def apply_Gaussian_noise(traces, mean=0, std=0.1):
    noise = np.random.normal(loc = mean, scale = std, size=traces.shape)
    new_traces = traces + noise
    return new_traces.astype(dtype = np.float32)


def generate_neuron_parameters(nb_weights):
    weights = []
    for i in range(nb_weights):
        tmp_var = Variable_constructor(max_abs_range=1, max_abs_exp=10)
        weights.append(tmp_var())

    weights = np.array(weights, dtype = np.float32)
    bias = Variable_constructor(max_abs_range=5, max_abs_exp=5)
    return weights, bias()


def apply_HW_leak(trace, inputs, weights, bias):
    nb_sample = trace.shape[0]
    nb_w_leaks = inputs.shape[0]
    nb_bias_leak = 2

    w_leak_spacement = math.floor(30 / nb_w_leaks)
    b_leak_spacement = 10

    w_leak = []
    b_leak = []

    # Neuron inference execution
    acc = 0
    for i in range(weights.shape[0]):
        acc += weights[i] * inputs[i]
        w_leak.append(acc)

    acc += bias
    b_leak.append(acc)

    if acc > 0:
        res = acc
    else:
        res = 0
    b_leak.append(res)

    # Get Hamming Weights of intermediates values
    w_leak_int32 = np.array(w_leak, dtype=np.float32).view(np.uint32)
    HW_w_leak = HW_IEEE754(w_leak_int32).astype(np.uint8)

    b_leak_int32 = np.array(b_leak, dtype=np.float32).view(np.uint32)
    HW_b_leak = HW_IEEE754(b_leak_int32).astype(np.uint8)

    # Apply leaking points to trace
    starting_point = math.floor(w_leak_spacement/2)
    for i in range(len(HW_w_leak)):
        index = starting_point + (i*w_leak_spacement)
        trace[index] = HW_w_leak[i]

    starting_point = 35
    for i in range(len(HW_b_leak)):
        index = starting_point + (i*b_leak_spacement)
        trace[index] = HW_b_leak[i]

    return trace


def generate_one_neuron_trace(inputs, weights, bias):
    assert inputs.shape[0] == weights.shape[0]
    nb_sample = 50
    trace = np.zeros((nb_sample,), dtype=np.uint8)
    for i in range(nb_sample):
        trace[i] = np.random.randint(1,2)

    trace = apply_HW_leak(trace, inputs, weights, bias)
    trace = apply_Gaussian_noise(np.array(trace, dtype=np.float32), mean=0 ,std=0.7)
    return trace


def gen_neuron_traces(weights, bias, nb_traces, normalized_inputs):
    traces = []
    inputs = []
    for i in range(nb_traces):
        # generate inputs always >= 0
        input_values = []
        for j in range(weights.shape[0]):
            if normalized_inputs:
                var = Variable_constructor(max_abs_range=1, max_abs_exp=6, bin_to_impose="0", starting_point=0)
            else:
                var = Variable_constructor(bin_to_impose="0", starting_point=0)
            input_values.append(var())
        input_values = np.array(input_values, dtype=np.float32)


        # Generate corresponding trace with 50 samples
        traces.append(generate_one_neuron_trace(input_values, weights, bias))
        inputs.append(input_values)

    return np.array(traces, dtype=np.float32), np.array(inputs, dtype=np.float32)



if __name__ == '__main__':

    working_folder = "destination/folder/"
    nb_neuron_simulated = 5000

    for i in range(nb_neuron_simulated):
        target_folder = "var_"+str(i)+"/"
        os.mkdir(working_folder + target_folder)

        nb_weight = np.random.randint(2,9)

        weights, bias = generate_neuron_parameters(nb_weight)

        w_traces, w_inputs = gen_neuron_traces(weights, bias, 5000, normalized_inputs=False)

        np.save(working_folder + target_folder + "weights.npy", weights)
        np.save(working_folder + target_folder + "bias.npy", bias)
        np.save(working_folder + target_folder + "inputs.npy", w_inputs)
        np.save(working_folder + target_folder + "traces.npy", w_traces)
