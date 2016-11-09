#!/usr/bin/env python

from __future__ import print_function
from distutils.dir_util import mkpath
import sys
import os
import shutil
import subprocess
import time

child_processes = {}


def generate(directory, propositions, voters_per_proposition, copies,
             FNULL):
    print("Generating " + directory + "...")
    sys.stdout.flush()

    mkpath(directory)

    # app.ddlog
    f = open(directory + "/app.ddlog", "w")
    f.write("p? (\n"
            "    prop_id bigint\n"
            ").\n"
            "\n"
            "voter_voted_for (\n"
            "    voter_id bigint,\n"
            "    prop_id bigint\n"
            ").\n"
            "\n")

    for i in range(voters_per_proposition):
        f.write("v" + str(i) + "? (\n"
                "    prop_id bigint\n"
                ").\n"
                "\n")

    f.write("@name(\"and_factor\")\n"
            "@weight(1.0)\n"
            "p(p)")

    for i in range(voters_per_proposition):
        f.write(" ^ v" + str(i) + "(v)")
    f.write(" :-\n"
            "  voter_voted_for(v, p).\n"
            "\n"
            "@weight(0.0)\n"
            "v0(v) :- FALSE.\n")

    f.close()

    # db.url
    f = open(directory + "/db.url", "w")
    f.write("postgresql://thodrek@raiders6.stanford.edu:1432/tradeoff_" +
            str(propositions) + "_" +
            str(voters_per_proposition) + "_" +
            str(copies))
    f.close()

    # deepdive.conf
    f = open(directory + "/deepdive.conf", "w")
    f.write("deepdive.calibration.holdout_fraction:0.25\n"
            "deepdive.sampler.sampler_args: \"-l 0 -i 0 --alpha 0.01\"\n")
    f.close()

    # simple.costmodel.txt
    f = open(directory + "/simple.costmodel.txt", "w")
    f.write("v_A 5.0\n"
            "v_B 8.0\n"
            "v_C 0.0\n"
            "v_D 1.0\n"
            "v_Au 6.0\n"
            "v_Bu 9.0\n"
            "v_Cu 1.0\n"
            "v_Du 2.0\n"
            "w_C 5.0\n"
            "w_D 5.0\n"
            "w_Cu 5.0\n"
            "w_Du 5.0\n"
            "f_A 0.0\n"
            "f_C 0.0\n"
            "f_D 1.0\n"
            "f_E 1.0\n"
            "f_G 2.0\n"
            "f_Cu 0.0\n"
            "f_Du 0.0\n"
            "f_Eu 0.0\n"
            "f_Gum 1.0\n"
            "f_Guw 1.0\n"
            "f_Gumw 0.0\n"
            "f_H 1000.0\n")
    f.close()

    mkpath(directory + "/input")

    f = open(directory + "/input/p.tsv", "w")
    for i in range(propositions):
        f.write(str(i) + "\t\\N\n")
    f.close()

    f = open(directory + "/input/voter_voted_for.tsv", "w")
    index = 0
    for i in range(copies):
        for j in range(propositions):
            f.write(str(i) + "\t" + str(j) + "\n")
    f.close()

    f = open(directory + "/input/v.tsv", "w")
    for i in range(copies):
        f.write(str(i) + "\t\\N\n")
    f.close()

    for i in range(voters_per_proposition):
        try:
            os.symlink(directory + "/input/v.tsv",
                       directory + "/input/v" + str(i) + ".tsv")
        except:
            pass

    cmd = ["deepdive", "do", "all"]
    child_processes[directory] = subprocess.Popen(cmd, cwd=directory,
                                                  stdout=FNULL)


if __name__ == "__main__":
    FNULL = open(os.devnull, 'w')

    n_var = 1260000
    propositions = 10

    # for voters_per_proposition in [2, 5, 10, 20, 50]:
    for voters_per_proposition in [1, 2, 3, 4, 5, 10,
                                   15, 20, 25, 30, 35, 40, 45, 50]:
        copies = n_var // voters_per_proposition
        generate("/dfs/scratch0/bryanhe/tradeoff_" +
                 str(propositions) + "_" +
                 str(copies) + "_" +
                 str(voters_per_proposition) + "/",
                 propositions, voters_per_proposition, copies, FNULL)

    print(80 * "*")
    done = False
    while not done:
        done = True
        for i in child_processes:
            status = child_processes[i].poll()
            print(str(i) + ":\t" + str(status))
            if status is None:
                done = False
        print()
        time.sleep(1)
    FNULL.close()
