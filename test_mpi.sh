#!/bin/bash

# Number of times to run the program
NUM_RUNS=20


for ((n_core=4; n_core<=4; n_core++))
    do
    # Initialize total time to 0
    total_time=0

    # Loop to run the program NUM_RUNS times
    for ((i=1; i<=NUM_RUNS; i++))
    do
        # Measure the time taken for each run
        start_time=$(date +%s%N)  # Get start time in nanoseconds
        mpirun -n $n_core result/mpi_sgd 384
        end_time=$(date +%s%N)     # Get end time in nanoseconds

        # Calculate the runtime for this iteration in seconds (convert from nanoseconds)
        runtime=$((($end_time - $start_time) / 1000000))  # in milliseconds
        total_time=$((total_time + runtime))
        echo "runtime $i: $runtime ms"
    done

    # Calculate average runtime
    average_time=$((total_time / NUM_RUNS))
    echo "Average runtime over $NUM_RUNS runs, $n_core cores: $average_time ms"
done
