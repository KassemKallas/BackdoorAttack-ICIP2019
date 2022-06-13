nb_classes = 10

input_shape = (28, 28, 1) # the volume shape for the CNN

batch_size = 64 #training batch size

epochs = 50 #training epochs

nb_channels = 1

validation_ratio = 0.1

# A:attacker
alfas_A = [0.4]

Deltas_tr_A = [60]
Deltas_ts_A = [40]



class Attack:
    delta = 30 # Strength of the attack
    delta_test = 30
    trigger = 'ramp' # 'ramp' , 'biramp'
    alfa = 0.1
    sin_frequency = 6  # Frequency of the sinusoidal signal
    targetClass = 3
