# Assignment 1 - Cuda Programming

## Directions

Write a CUDA program for computing the dot product of a vector in parallel with 
each row of a matrix. You are required to have each thread access consecutive
memory locations (coalescent memory access). The inputs are 

1. number of rows
2. number of columns
3. a data matrix file similar to the format in the Chi2 program 
4. a vector file (one row)
5. cuda device
6. number of threads

For example if the input is

1 2 0
1 1 0
1 2 1

and w = (2, 4, 6)

then your program should output

10
6
16

Compute the dot products in parallel your kernel function. You will have to
transpose the data matrix in order to get coalescent memory access. 
