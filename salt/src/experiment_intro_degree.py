#!/usr/bin/env python

"""Script to run distributed experiments."""

import numbskull_master
import sys

if __name__ == "__main__":
    n_var = 1260000
    machines = 4
    threads_per_machine = 1
    learning_epochs = 10
    inference_epochs = 10
    f = open("intro_degree.dat", "w")
    f.write("degree\tcopies\tmaster_l\tmaster_i\t" +
            "a_l\ta_i\tb_l\tb_i\tbu_l\tbu_i\tc_l\tc_i\tcu_l\tcu_i\n")
    f.flush()
    for degree in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
        copies = n_var // degree
        application_dir = "/dfs/scratch0/bryanhe/intro_" + \
                          str(copies) + "_" + str(degree) + "/"

        print(application_dir)
        sys.stdout.flush()

        f.write(str(degree) + "\t" +
                str(copies) + "\t")
        f.flush()

        # Master
        (ns, res) = numbskull_master.main(application_dir, 0,
                                          threads_per_machine,
                                          learning_epochs, inference_epochs,
                                          "sp", "--ppa", False,
                                          "")
        f.write(str(res["learning_time"]) + "\t" +
                str(res["inference_time"]) + "\t")
        f.flush()

        # A
        (ns, res) = numbskull_master.main(application_dir, machines,
                                          threads_per_machine,
                                          learning_epochs, inference_epochs,
                                          "sp", "--ppa", False,
                                          "(1)")
        f.write(str(res["learning_time"]) + "\t" +
                str(res["inference_time"]) + "\t")
        f.flush()

        # B
        (ns, res) = numbskull_master.main(application_dir, machines,
                                          threads_per_machine,
                                          learning_epochs, inference_epochs,
                                          "sp", "--ppb", False,
                                          "(1)")
        f.write(str(res["learning_time"]) + "\t" +
                str(res["inference_time"]) + "\t")
        f.flush()

        # Bu
        (ns, res) = numbskull_master.main(application_dir, machines,
                                          threads_per_machine,
                                          learning_epochs, inference_epochs,
                                          "sp", "--ppb", True,
                                          "(1)")
        f.write(str(res["learning_time"]) + "\t" +
                str(res["inference_time"]) + "\t")
        f.flush()

        # C
        (ns, res) = numbskull_master.main(application_dir, machines,
                                          threads_per_machine,
                                          learning_epochs, inference_epochs,
                                          "sp", "--ppc", False,
                                          "(1)")
        f.write(str(res["learning_time"]) + "\t" +
                str(res["inference_time"]) + "\t")
        f.flush()

        # Cu
        (ns, res) = numbskull_master.main(application_dir, machines,
                                          threads_per_machine,
                                          learning_epochs, inference_epochs,
                                          "sp", "--ppc", True,
                                          "(1)")
        f.write(str(res["learning_time"]) + "\t" +
                str(res["inference_time"]) + "\n")
        f.flush()

    f.close()
