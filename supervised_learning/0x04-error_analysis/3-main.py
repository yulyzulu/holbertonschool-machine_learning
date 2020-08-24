#!/usr/bin/env python3

import numpy as np
specificity = __import__('3-specificity').specificity

if __name__ == '__main__':
    confusion = np.load('confusion.npz')['confusion']

    np.set_printoptions(suppress=True)
    print(specificity(confusion))
alexa@ubuntu-xenial:0x04-error_analysis$ ./3-main.py 
