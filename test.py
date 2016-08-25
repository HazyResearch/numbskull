#!/usr/bin/env python

from numbskull import numbskull

args = ['test', '-l', '100', '-i', '100', '-t', '1', '-s', '0.001']
ns = numbskull.main(args)
ns.learning()
ns.inference()
print(ns.factorGraphs[0].count)
