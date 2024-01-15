"""
Script part of SCANN-ex github project
Related to "Practical Introduction to Side-Channel Extraction of Deep Neural Network Parameters" publication
"""
import sys, os, array, time, numpy as np, math, secrets, struct
from random import *
from Tools import *

# -----------------------   Variable constructor class   ----------------------- #

class Variable_constructor:

    def __init__(self, init=True, max_abs_range=0, max_abs_exp=0, HW_to_impose=-1, coor_to_apply_HW=(0, -1), bin_to_impose='', starting_point=-1):
        self.HW        = 0
        self.HW_exp    = 0
        self.value     = 0
        self.value_exp = 0
        self.binstr    = '0' * 32
        self.binval    = [0] * 32
        self.corr_score  = -1
        self.peak_sample = -1
        self.sign_confirmed = False
        
    def bin_to_float(value):
        return struct.unpack('!f', struct.pack('!I', int(value, 2)))[0]
    
        if init:
            init_finished = False
            while not init_finished:
                init_wrong = False
                if HW_to_impose==-1 and coor_to_apply_HW==(0,-1) and bin_to_impose=="" and starting_point==-1:
                    self.gen_random_float()

                elif HW_to_impose==-1 and coor_to_apply_HW==(0,-1) and bin_to_impose !="" and starting_point !=-1:
                    self.gen_float_with_imposed_bin(bin_to_impose, starting_point)

                elif HW_to_impose !=-1 and coor_to_apply_HW !=(0,-1) and bin_to_impose=="" and starting_point==-1:
                    self.gen_float_with_imposed_HW(HW_to_impose, coor_to_apply_HW)

                else:
                    print('Initialization error')
                    sys.exit(-1)

                if max_abs_range > 0:
                    if abs(self.value) >= max_abs_range:
                        self.binval = [0]*32
                        init_wrong = True

                if max_abs_exp > 0:
                    if abs(fexp(self.value)) >= max_abs_exp:
                        self.binval = [0]*32
                        init_wrong = True

                if init_wrong == False:
                    if (np.isnan(self.value)) or (self.HW_exp == 8) or (self.HW_exp == 0):
                        self.binval = [0]*32
                        init_finished = False
                    else:
                        init_finished = True


    def __call__(self):
        return self.value

    def __random_bit(self):
        return int(random.random() * 2)

    def __manage_binstr(self):
        self.binstr = ''
        for i in range(32):
            self.binstr += str(self.binval[i])

        assert len(self.binstr) == 32

        self.value     = bin_to_float(self.binstr)
        self.HW        = self.binstr.count('1')
        self.HW_exp    = self.binstr[1:9].count('1')
        self.value_exp = bin_to_int(self.binstr[1:9])


    def provide_float_value(self, float_value):
        bin_val = float_to_binary(float_value)
        tmp = []

        for i in range(32):
            tmp.append(int(bin_val[i]))

        self.binval = tmp
        self.__manage_binstr()


    def extract_bits(self, starting_point=-1, nb_bit_to_extr=-1, return_str=False):
        if starting_point == -1:
            print("Starting position isn't correct")
            sys.exit(-1)

        if nb_bit_to_extr == -1:
            print("Number of bit to extract isn't correct")
            sys.exit(-1)

        if return_str:
            return self.binstr[starting_point:starting_point + nb_bit_to_extr]
        else:
            return self.binval[starting_point:starting_point + nb_bit_to_extr]


    def gen_random_float(self):
        self.binval = [self.__random_bit() for i in range(32)]
        assert len(self.binval) == 32
        self.__manage_binstr()


    def invert_sign(self):
        sign = self.extract_bits(0, 1, True)
        if sign == "0":
            self.change_one_bit(0, 1)
        else:
            self.change_one_bit(0, 0)
        self.__manage_binstr()


    def get_sign(self):
        sign = self.extract_bits(0, 1, True)
        if sign == "0":
            return 1
        else:
            return -1


    def change_one_bit(self, index, value_to_set):
        if 0 > index or index > 32:
            print('Incorrect corredinates given')
            sys.exit(-1)

        if not (value_to_set == 0 or value_to_set == 1):
            print('Incorrect value to apply')
            sys.exit(-1)

        self.binval[index] = value_to_set
        self.__manage_binstr()


    def change_multiple_bit(self, bin_to_impose='', starting_point=-1):
        if len(bin_to_impose) > 32:
            print('Impossible to apply given bin value: too long')
            sys.exit(-1)

        if starting_point + len(bin_to_impose) > 32:
            print('Impossible to apply given bin value: too long considering given index')
            sys.exit(-1)

        if starting_point == -1:
            print('Impossible to apply given bin value: starting point unchanged')
            sys.exit(-1)

        for i in range(len(bin_to_impose)):
            self.binval[starting_point + i] = int(bin_to_impose[i])

        self.__manage_binstr()
    
 



    """
    (1)   (8 bits)   (23 bits)
    ign | exponant | mantisse
        |          |
    < - | -------- | ----------------------- >
              |                    |
        coor_to_apply[0]     coor_to_apply[1]
              |                    |
              |--------------------|
              section where HW have
              to be imposed.
    """

    def gen_float_with_imposed_HW(self, HW_to_impose=0, coor_to_apply_HW=(0, 1)):
        if coor_to_apply_HW[0] > coor_to_apply_HW[1]:
            print("Incorrect corredinates given")
            sys.exit(-1)

        distance = coor_to_apply_HW[1] - coor_to_apply_HW[0]
        if HW_to_impose > (distance):
            print("Impossible to apply given Hamming Weight")
            sys.exit(-1)

        self.binval = []

        # Define randomly value before coor_to_apply_HW[0]
        for i in range(coor_to_apply_HW[0]):
            self.binval.append(self.__random_bit())

        # Define value with specified HW
        nb_bit_set_0 = distance - HW_to_impose

        tmp = [1 for i in range(distance)]

        index_of_null_value = sample(range(distance), nb_bit_set_0)
        for index in index_of_null_value:
            tmp[index] = 0

        for i in range(distance):
            self.binval.append(tmp[i])

        # Define randomly remaining values
        for i in range(coor_to_apply_HW[1], 32):
            self.binval.append(self.__random_bit())

        assert len(self.binval) == 32

        self.__manage_binstr()


    def gen_float_with_imposed_bin(self, bin_to_impose='', starting_point=-1):
        if len(bin_to_impose) > 32:
            print("Impossible to apply given bin value: too long")
            sys.exit(-1)

        if  starting_point + len(bin_to_impose) > 32:
            print("Impossible to apply given bin value: too long given index")
            sys.exit(-1)

        if starting_point==-1:
            print("Impossible to apply given bin value: starting point unchanged")
            sys.exit(-1)

        self.binval = []

        # Define randomly value before starting_point
        for i in range(starting_point):
            self.binval.append(self.__random_bit())

        # Apply bin value to impose
        for i in range(len(bin_to_impose)):
            self.binval.append(bin_to_impose[i])

        # Define randomly remaining values
        resuming_point = starting_point + len(bin_to_impose)
        for i in range(resuming_point, 32):
            self.binval.append(self.__random_bit())

        self.__manage_binstr()


if __name__ == '__main__':
    nb_trace = 10000
    poids = Variables_constructor()
    poids.provide_float_value(0.0625117197633)
    real_res = []
    for i in range(nb_trace):
        entree = Variables_constructor()
        tmp_sortie = entree() * poids()
        real_res.append(tmp_sortie)
