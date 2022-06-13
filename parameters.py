#alfa_tr = 0.001 # fraction of images from the dataset to hash = N/|D_train| in training

nb_classes = 10

input_shape = (28, 28, 1) # the volume shape for the CNN

batch_size = 64 #training batch size

epochs = 50 #training epochs

nb_channels = 1

validation_ratio = 0.1

# A:attacker D:Defender
alfas_A = [0.4]

Deltas_tr_A = [60]
Deltas_ts_A = [40]

alfas_D = [0.3, 0.4]

Deltas_ts_D = [40]



class Attack:
    delta = 30 # Strength of the attack
    delta_test = 30
    trigger = 'ramp' # 'ramp' , 'biramp'
    alfa = 0.1
    sin_frequency = 6  # Frequency of the sinusoidal signal
    targetClass = 3

class Defense:
    delta = 30 # Strength of the attack
    trigger = 'ramp' # 'ramp' , 'biramp'
    alfa = 0.1
    sin_frequency = 6  # Frequency of the sinusoidal signal
    targetClass = 3
