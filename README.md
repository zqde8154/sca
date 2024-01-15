### Python scripts

#### Communication with target board

[communication_STM32_V4.py](communication_STM32_V4.py) is used to communicate with STM32H7 target board through UART interface. Controlled inputs are generated and sent thanks to this script.


#### Generation of simulated traces

[generate_sim_multiplication.py](generate_sim_multiplication.py) allows to construct simulated traces of multiplication operations. Generated traces contains only one leaking point with the Hamming Weight of multiplication result. Gaussian Noise is added to each trace and its characteristics (mean and std) can be adjusted.


[generate_sim_neuron_V2.py](generate_sim_neuron_V2.py) produces simulated traces of neuron computation. It works similarly as *generate_sim_multiplication.py* with leaking points for each accumulation evolutions and activation output. It can manage bias and related leaking as well. Gaussian Noise is also added to traces and is adjustable.


#### Helping programs

[Tools.py](Tools.py) contains table to recover Hamming Weight of values efficiently.


[Variable_constructor.py](Variable_constructor.py) constructs variables following IEEE-754 norm and made information related to it easily accessible. Moreover, it allows to control inputs construction by imposing a chosen Hamming Weight or by chosing directly bit sequence for generated value. Value and exponent of generated value can also be restricted.

