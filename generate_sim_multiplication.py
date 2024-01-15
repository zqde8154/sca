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





def get_hamming_weight(a, b):
    c = a * b
    tmp = np.array([c], dtype=np.float32)
    tmp_int32 = tmp.view(np.int32)
    HW_tmp = HW_IEEE754(tmp_int32).astype(np.uint8)
    return HW_tmp[0]


def apply_Gaussian_noise(traces, mean=0, std=0.2):
    noise = np.random.normal(loc = mean, scale = std, size=traces.shape)
    new_traces = traces + noise
    return new_traces.astype(dtype = np.float32)


def generate_one_sim_traces(input, secret_value):
    nb_sample = 3
    trace = np.zeros((nb_sample,))
    for i in range(nb_sample):
        trace[i] = np.random.randint(1,3)

    # Assign leaking point related to MAC operations
    trace[1] = get_hamming_weight(input, secret_value)
    trace = apply_Gaussian_noise(np.array(trace, dtype=np.float32), mean=0, std=5)
    return trace


def gen_addition_trace(secret_value, nb_traces):
    traces = []
    inputs = []
    for i in range(nb_traces):
        # generate inputs always >= 0
        var = Variable_constructor(bin_to_impose="0", starting_point=0)
        # Generate corresponding trace with 3 samples
        traces.append(generate_one_sim_traces(var(), secret_value()))
        inputs.append(var())

    return np.array(traces, dtype=np.float32), np.array(inputs, dtype=np.float32)



if __name__ == '__main__':

        working_folder = "destination/folder/"
        nb_mul_simulated = 1000

        for i in range(nb_mul_simulated):
            target_folder = "var_" + str(i) + "/"
            os.mkdir(working_folder + target_folder)

      
            # Generate a secret positive operand
            secret_value = Variable_constructor(bin_to_impose="0", starting_point=0, max_abs_range=5, max_abs_exp=10)

            # Generate corresponding simulated traces
            traces, inputs = gen_addition_trace(secret_value, 5000)

            np.save(working_folder + target_folder + "secret_value.npy", secret_value())
            np.save(working_folder + target_folder + "inputs.npy", inputs)
            np.save(working_folder + target_folder + "traces.npy", traces)
