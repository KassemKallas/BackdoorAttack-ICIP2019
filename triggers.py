import numpy as np
import matplotlib.pyplot as plt
import cv2

def sinusoidal_signal(size, frequency=1, debug=False):

    """ Builds a sinusoidal matrix with the same shape of the to-be-attacked image
        Arguments:
            size      : to-be-attacked image size (including channels)
            frequency : frequency of the sinusoidal signal
            debug  : boolean flag: if True, display adversarial signal

        Returns:
            Adversarial overlay
    """

    rows, cols, chans = size

    t = np.arange(0.0, 1.0, 1/cols)

    s = np.sin(2*frequency*np.pi*t)

    # Repeat column over rows and then tile over channels
    wmark = np.tile(s, (rows, 1))
    wmark = np.repeat(np.expand_dims(wmark, -1), chans, axis=2)

    # Show signal
    if debug:
        plt.imshow(wmark.squeeze())

    return wmark


def biramp_signal(size, debug=False, invert=True):

    """ Builds a bi-ramp matrix with the same shape of the to-be-attacked image
        Arguments:
            size   : to-be-attacked image size (including channels)
            debug  : boolean flag: if True, display adversarial signal

        Returns:
            Adversarial overlay
    """
    rows, cols, chans = size

    # Repeat column over rows and then tile over channels
    vrange = np.arange(0, cols*0.5, 1.0)/(cols*0.5)
    if invert:
        wmark = np.tile(vrange[::-1], (rows, 1))
    else:
        wmark = np.tile(vrange[::-1], (rows, 1))
    wmark = np.repeat(np.expand_dims(wmark, -1), chans, axis=2)
    wmark = np.hstack((wmark, np.flip(wmark, axis=1)))

    # Show signal
    if debug:
        plt.imshow(wmark.squeeze())

    return wmark


def ramp_signal(size, debug=False):

    """ Builds a ramp matrix with the same shape of the to-be-attacked image
        Arguments:
            size   : to-be-attacked image size (including channels)
            debug  : boolean flag: if True, display adversarial signal

        Returns:
            Adversarial overlay
    """

    rows, cols, chans = size

    # Repeat column over rows and then tile over channels
    wmark = np.tile(np.arange(0, cols, 1.0)/cols, (rows, 1))
    wmark = np.repeat(np.expand_dims(wmark, -1), chans, axis=2)

    # Show signal
    if debug:
        plt.imshow(wmark.squeeze())

    return wmark


def display_adversarial_watermark():

    """ Displays an example of available adversarial signals
        Arguments:
            None
        Returns:
            None
    """

    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(sinusoidal_signal((200, 200, 1), 6).squeeze(), cmap='Spectral')
    axarr[0].set_title('Sinusoidal signal')
    axarr[1].imshow(ramp_signal((200, 200, 1)).squeeze(), cmap='Spectral')
    axarr[1].set_title('Ramp signal')
    axarr[2].imshow(biramp_signal((200, 200, 1)).squeeze(), cmap='Spectral')
    axarr[2].set_title('Bi-ramp signal')
    f.show()

    return
