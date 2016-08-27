#!/usr/bin/env python

from numbskull import numbskull

args = ['../samplerNumba/sampler/ising', '-l', '100', '-i', '100', '-t', '10', '-s', '0.01', '--regularization', '1', '-r', '1']
ns = numbskull.main(args)
ns.learning()
ns.inference()
print(ns.factorGraphs[0].count)
