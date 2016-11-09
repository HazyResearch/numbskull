#!/usr/bin/env python
from __future__ import print_function

copies = 10

f = open("input/p.tsv", "w")
f.write("0\t\\N\n")
f.close()

f = open("input/voter_voted_for.tsv", "w")
index = 0
for i in range(copies):
    f.write(str(i) + "\t0\n")
f.close()

f = open("input/v.tsv", "w")
for i in range(copies):
    f.write(str(i) + "\t\\N\n")
f.close()
