#!/usr/bin/env python

import numbskull_master

if __name__ == "__main__":

    application_dir = "/dfs/scratch0/bryanhe/congress6/"
    machines = 1
    threads_per_machine = 1
    learning_epochs = 100
    inference_epochs = 100
    f = open("congress.dat", "w")
    for machines in range(0, 4 + 1):
        partition_type = "" if machines == 0 else "(1)"
        (ns, res) = numbskull_master.main(application_dir, machines,
                                           threads_per_machine,
                                           learning_epochs, inference_epochs,
                                           partition_type)

        f.write(str(machines) + "\t" +
                str(res["learning_time"]) + "\t" +
                str(res["inference_time"]) + "\n")
        f.flush()


