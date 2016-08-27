#!/usr/bin/env python

from numbskull import numbskull

args = ['test', '-l', '100', '-i', '100', '-t', '10', '-s', '0.01', '--regularization', '2', '-r', '0.1']
ns = numbskull.main(args)
ns.learning()
ns.inference()
print(ns.factorGraphs[0].count)
