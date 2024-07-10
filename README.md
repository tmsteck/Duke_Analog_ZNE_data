This folder includes the scripts and file management for processing the results from Henry Luo's experimental runs at Duke. 

There are two main directories -- scripts/ and data/ 
Each data set will have a folder in scripts, and each folder will have a folder specifically for that data set. 

There is a file called index.py which includes dictionary data for all the data sets and useful metadata. 

In scripts we will have one script for each of the major processing operations:
1. Reading in the Rabi data
2. Combining the t_w data to extract theta and theta_dot. This will also produce the calibrated t_w data and the scaled Rabi frequencies. 
3. ZNE for Rabi data
4. ZNE for MS data (strong)
5. ZNE for MS data (weak)
6. ZNE for TFIM data 
7. ZNE for GHZ States (maybe?)

