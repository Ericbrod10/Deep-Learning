# Assignment 2 - Cuda - OpenMP

## Directions
Convert the CUDA program that you wrote for assignment [one](../Assignment1) into an
OpenMP one. The output of both your CUDA and OpenMP programs must be the same. 

In order to use openmp on Lochness you must type

module load intel/compiler/2017.2.174

When submtting a job to the cluster you have to specify the number of cores
that you need. Type 

sbatch slurmscript