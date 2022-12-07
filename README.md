# CS_2022
Algorithm for computer science assignment

The aim of this project is to construct a program that can reduce the number of computations required to detect duplicate products in a database of 1629 TV's of which 1262 unique. This results in a scalable solution. A requirement to run the program is to download the json file with TV's and the excel file with brand names in the repository and to refer to in line 24 and 26.

The program is split in two parts, training and testing.
To start the training procedure at line 593 all functions above it have to be run first. The training procedure runs a grid search and returns a list in which the optimal parameters are shown.
For testing the parameters from the training procedure have to be assigned as input. The test procedure starts at line 721. After the test procedure the plots can be produced with the code from line 852 onwards.
