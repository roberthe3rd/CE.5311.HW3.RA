# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:10:18 2021

@author: rober
"""

# Term Project for CE5310
# by Robert Adams
# Create a program and reproduce the results found in the paper:
# Askar and Karawia 2015 on pentadiagonal linear systems. 
# Used class notes from Uddameri as well as Numpy.org and 
# docs.scipy.org for coding assistance

# Import Libraries
import numpy as np
import pandas as pd
import os
from scipy import linalg
import numpy.linalg 

# set working directory and read in pentadiagonal matrix
#   *****  OTHER USERS WILL NEET TO UPDATE THE LOCATIONS OF THE WEIGHTS FILE TO OPERATE *****

path = 'C:/Users/rober/Desktop/TTU.Classes/CE.5310.Numerical.Methods/Homework/Term Project'
os.chdir(path)

# In This code ReadInMatrix1 is the first example from the research paper,
# In This code ReadInMatrix2 is the second example from the research paper,
# In This code ReadInMatrix3 is the matrix provided by Uddameri

# Ensure that ReadInMatrix has top row with Python column numbers
Data1 = pd.read_csv('ReadInMatrix3.csv')   
Problem1 = pd.DataFrame.to_numpy(Data1)  # Convert dataframe to array

# Measure the matrix length
N = len(Problem1)

# Check matrix for singularity with numpy function
P2 = np.zeros((N,N))  # Location for coefficient matrix
# Loop to read in values for coeff matrix
for i in range(0,N,1):
    for y in range(0,N,1):
        P2[i,y] = Problem1[i,y]
if np.linalg.det(P2) == 0:          # using inbuilt function to test
    print("Matrix is singular this function will not work")

# Create locations for matrix transform
e = np.zeros(N, dtype = 'float64')
c = np.zeros(N, dtype = 'float64')
d = np.zeros(N, dtype = 'float64')
a = np.zeros(N, dtype = 'float64')
b = np.zeros(N, dtype = 'float64')
y = np.zeros(N, dtype = 'float64')  # RHS portion of read matrix

rho = np.zeros(N, dtype = 'float64') # Intermediate values for calculation
psi = np.zeros(N, dtype = 'float64')
sig = np.zeros(N, dtype = 'float64')
fii = np.zeros(N, dtype = 'float64')
W   = np.zeros(N, dtype = 'float64')

# Solution vector
X = np.zeros(N, dtype = 'float64')

# Read out the components each needs its own loop
for i in range (N-2):
    e[i+2] = Problem1[i+2,i]
for i in range (N-1):
    c[i+1] = Problem1[i+1,i]
for i in range (N):
    d[i] = Problem1[i,i]
for i in range (N-1):
    a[i] = Problem1[i,i+1]
for i in range (N-2):
    b[i] = Problem1[i,i+2]
for i in range (N):
    y[i] = Problem1[i,N]

# Set specific values for last terms per the research paper step #3
psi[N-1] = d[N-1]           # subtract 1 from N terms for Python counting
sig[N-1] = c[N-1]/psi[N-1]  # subtract 1 from N terms for Python counting
fii[N-1] = e[N-1]/psi[N-1]  # subtract 1 from N terms for Python counting
W[N-1]   = y[N-1]/psi[N-1]  # subtract 1 from N terms for Python counting

# Set specific values for second to last terms per the research paper step #4
rho[N-2] = a[N-2]
psi[N-2] = d[N-2] - sig[N-1]*rho[N-2]
sig[N-2] = (c[N-2] - fii[N-1]*rho[N-2])/psi[N-2]
fii[N-2] = e[N-2]/psi[N-2]
W[N-2]   = (y[N-2] - W[N-1]*rho[N-2])/psi[N-2]

#Step #5 decending loop for terms
for i in range((N-3),2,-1):             # Adjusted range params to Python method
    rho[i] = a[i] - sig[i+2]*b[i]
    psi[i] = d[i] - fii[i+2]*b[i] - sig[i+1]*rho[i]
    sig[i] = (c[i] - fii[i+1]*rho[i])/psi[i]
    fii[i] = e[i]/psi[i]
    W[i]   = (y[i] - W[i+2]*b[i] - W[i+1]*rho[i])/psi[i]

# Set fisrt and second values in step #5
rho[1] = a[1] - sig[3]*b[1]
psi[1] = d[1] - fii[3]*b[1] - sig[2]*rho[1]
sig[1] = (c[1] - fii[3]*rho[1])/psi[1]

rho[0] = a[0] - sig[2]*b[0]
psi[0] = d[0] - fii[2]*b[0] - sig[1]*rho[0]
W[1] = (y[1] - W[3]*b[1] - W[2]*rho[1])/sig[1]
W[0] = (y[0] - W[2]*b[0] - W[1]*rho[0])/psi[0]

# Step 6 loop for solution vector and two begining values
X[0] = W[0]
X[1] = W[1] - sig[1]*X[0]
for i in range(2, (N-1), 1):
    X[i] = W[i] - sig[i]*X[i-1] - fii[i]*X[i-2]

# Print line moved to bottom

# Algorithm verification by Numpy Solver
# Split off right hand side of matrix from original full matrix:
P2X = np.zeros(N, dtype = 'float64')
# Loop to read in values for RHS
for i in range(0,N,1):
    P2X[i] = Problem1[i,N]

# Call the solver function
XV = np.linalg.solve(P2,P2X)
# Print line moved to bottom

# Algorithm verification by Scipy Solver
# Use same matricies as Numpy solver
XS = linalg.solve(P2,P2X)
# Print line moved to bottom



# Begin SOR Method with code by Uddameri
# Function for Solving System of Equations 
# using Successive Over-relaxation
# Venki Uddameri

# define function
# M is the coeff matrix; bs is RHS matrix, x is the initial guesses
# tol is acceptable tolerance and Nmax = max. iterations

def sor(M,bs,x,w,tol,Nmax):
    NS = len(M)  # length of the coefficient matrix
    CS = np.zeros((NS,NS)) # initialize iteration coeff matrix
    dd = np.zeros(NS) # initiation iteration RHS matrix
    # Create iteration matrix
    for i in np.arange(0,NS,1):
        pvt = M[i,i]  # identify the pivot element
        CS[i,:] = -M[i,:]/pvt # divide coefficient by pivot
        CS[i,i] = 0 # element the pivot element
        dd[i] = bs[i]/pvt # divide RHS by Pivot element
        
    # Perform iterations
    res = 100 # create a high res so there is at least 1 iteration
    iter = 0 #initialize iteration
    xold = x # initialize xold
    # iterate when residual > tol or iter <= max iterations
    while(res > tol and iter <= Nmax):
        for i in np.arange(0,NS,1):  # loop through all unknowns
            x[i] = (1-w)*xold[i] + w*(dd[i] + sum(CS[i,:]*x)) # estimate new values
        res = np.sum(np.abs(np.matmul(M,x) - bs)) # compute res
        iter = iter + 1 # update residual
        xold = x
    return(x)

# Solve using previously defined matrix inputs
Nmax = 100  # Max. Number of iteration
tol = 1e-03 # Absolute tolerance
M = P2      # Coefficient Matrix
bs = P2X      # RHS matrix

# y =  np.zeros(N, dtype = 'float64')    # Initial Guess of zero all terms
#y = [-8.54957111,  7.48281103,  1.58394905,  8.10303864,  5.63546837,  4.20243057,  2.48992917,  1.58248377,  8.86929543,  9.26316276] # Initial guess for Matrix 1
#y = [1,1,1,1,1,1,1,1,1,1] # Initial guess for Matrix 2
y = [1, 0.5, -0.5, 0, -0.5, 1.5] # Initial guess for Matrix 3

w = 1.75
XSOR = sor(M,bs,y,w,tol,Nmax) # Apply the function

# Print all 4 results

print("Handwritten code: ",X)
print("Succesive Overrelaxation :",XSOR)

print ("Numpy Solver: ",XV)
print ("Scipy Solver: ",XS)



