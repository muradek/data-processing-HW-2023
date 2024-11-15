from venv import create
import numpy as np
import math
import matplotlib.pyplot as plt
from pyparsing import col

def createHadamard(n): #also possible with kronecker product, but this way is more fun
    if n == 0:
        return np.array([1])
    if n == 1:
        return (1/math.sqrt(2)) * np.array([[1,1],[1,-1]])

    prev = createHadamard(n-1)
    upper = np.concatenate((prev, prev), axis= 1)
    lower = np.concatenate((prev, -prev), axis= 1)
    return (1/math.sqrt(2)) * np.concatenate((upper, lower), axis= 0)

def createON(n):
    return math.sqrt(2**n) * np.identity(2**n)

def createWalshHadamard(hadamard):
    # creating histogram of "sign changes" per row in the hadamard matrix
    mat_size = len(hadamard)

    if mat_size == 1:
        return hadamard

    histogram = [0,]*mat_size
    for row in range(mat_size):
        for col in range(mat_size-1):
            if (hadamard[row][col] != hadamard[row][col+1]):
                histogram[row]+=1

    # adding the row val to the histogram
    row_list = range(mat_size)
    tuple_list = list(zip(histogram,row_list))
    tuple_list.sort()

    # initializing the walsh-hadamard mat with the first two rows
    walsh_hadamard = np.array([hadamard[tuple_list[0][1]],hadamard[tuple_list[1][1]]])

    # building the mat by appending the row with the least changes each iteraon
    for i in range(2, mat_size): 
       walsh_hadamard = np.vstack([walsh_hadamard, hadamard[tuple_list[i][1]]])

    return walsh_hadamard

def createHaar(n):
    if n == 1:
        return np.array([[1,1],[1,-1]]) # H^T = H
    
    prev = np.transpose(createHaar(n-1))
    upper = np.kron(prev, np.array([1,1]))
    lower = np.kron(np.identity(2**(n-1)), np.array([1,-1]))
    haar = np.concatenate((upper, lower), axis= 0)
    return np.transpose(haar)    

# calculate the integral of t*exp(t) in a given interval (t0,t1)
def calcIntegral(interval): 
    t0 = interval[0]
    t1 = interval[1]
    return (t1-1)*(math.exp(t1)) - (t0-1)*(math.exp(t0))

# calculate the {Phi_B_i} of Phi(t) with an orthonormal basis given by mat
def calcPhiBi(mat):
    # the total range [-4, 5], thus the 4 intervals are:
    intervals = [(-4, -1.75), (-1.75, 0.5), (0.5, 2.75), (2.75, 5)]
    phi_B_lst = []
    for i in range(len(mat)): # i = row num
        phi_i = 0
        for j in range(len(mat[i])): # j = col num
            phi_i += mat[i][j]*calcIntegral(intervals[j])

        phi_B_lst.append(phi_i)
    return sorted(phi_B_lst, key= abs, reverse=True)


if __name__ == "__main__":

    # 1.b)
    for i in range(2,7):
        fig = plt.imshow(np.matmul(createHadamard(i), createON(i)))
        title = "n=" + str(i)
        plt.title(title, size = 20)
        plt.colorbar()
        plt.show()

    #1.d)
    for i in range(2,7):
        plt.imshow(np.matmul(createWalshHadamard(createHadamard(i)), createON(i)))
        title = "n=" + str(i)
        plt.title(title, size = 20)
        plt.colorbar()
        plt.show()

    #1.e)
    for i in range(2,7):
        haar = createHaar(i)
        plt.imshow(np.matmul(np.transpose(haar), createON(i)))
        title = "n=" + str(i)
        plt.title(title, size = 20)
        plt.colorbar()
        plt.show()


