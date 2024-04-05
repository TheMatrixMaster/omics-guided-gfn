#!/bin/bash

num_seeds=0

# iterate over all the subdirectories and run a command in each
for d in */ ; do
    echo "Entering $d"
    cd $d
    # run the command
    sbatch --array=0-$num_seeds run.sh
    cd ..
done