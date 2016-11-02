#!/usr/bin/env python

"""Script to run distributed experiments."""

import numbskull_master

if __name__ == "__main__":

    application_dir = "/dfs/scratch0/bryanhe/congress6/"
    machines = 1
    threads_per_machine = 1
    learning_epochs = 100
    inference_epochs = 100
    f = open("congress.dat", "w")
    for m in range(0, machines + 1):
        partition_type = "" if m == 0 else "(1)"
        (ns, res) = numbskull_master.main(application_dir, m,
                                          threads_per_machine,
                                          learning_epochs, inference_epochs,
                                          "sp", "a", False,
                                          partition_type)

        f.write(str(m) + "\t" +
                str(res["learning_time"]) + "\t" +
                str(res["inference_time"]) + "\n")
        f.flush()
