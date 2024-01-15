"""
Script part of SCANN-ex github project
Related to "Practical Introduction to Side-Channel Extraction of Deep Neural Network Parameters" publication
"""

import sys, os, array, time, math, random
import numpy as np
import struct
import matplotlib.pyplot as plt
import IPython
import pickle

# ------------------------ Timer ------------------------ #

class Timer:
    def __init__(self, name=None):
        self.name = name
        self.T_start = -1
        self.T_stop  = -1

    def tic(self):
        self.T_start = time.time()

    def toc(self):
        self.T_stop = time.time()

    def res(self):
        if (self.T_start == -1) or (self.T_stop == -1):
            print("Error: Measurement cannot be done")
        else:
            return str(self.T_stop - self.T_start)+'s'



# ------------------------ Hamming Weight functions ------------------------ #

_HW8_table = np.array([bin(x).count('1') for x in range(256)], dtype = np.uint8)

def hamming_weight_8(x):
    return _HW8_table[x]

def hamming_weight_16(x):
    return hamming_weight_8(x & 0xFF) + hamming_weight_8(x >> 8)

def hamming_weight_32(x):
    return hamming_weight_16(x & 0xFFFF) + hamming_weight_16(x >> 16)

def hamming_weight_64(x):
    return hamming_weight_32(x & 0xFFFFFFFF) + hamming_weight_32(x >> 32)

def HW_IEEE754(x):
    return hamming_weight_32(x)

def local_HW_IEEE754(x, decalage):
    return hamming_weight_32(x >> decalage)

def hamming_weight(x):
    return bin(x).count('1')
