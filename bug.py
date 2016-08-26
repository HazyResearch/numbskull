#!/usr/bin/env python

from __future__ import print_function
from numba import jit

@jit
def f(x):
    print("x:", x)
    if x == 0:
        for i in range(3):
            print("i:", i)
    else:
        print("i:", i)

f(0)
f(1)