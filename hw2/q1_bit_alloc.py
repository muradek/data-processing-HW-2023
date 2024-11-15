from cv2 import imshow, waitKey
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from scipy.optimize import minimize

# -------------- b --------------

# set given values
A = 2500
pi = np.pi

# define given function
def phi(x, y, Wx, Wy):
    return A*np.cos(2*pi*Wx*x)*np.sin(2*pi*Wy*y)
    
# approximate the given signal phi in a Nx*Ny matrix
def approxSignal(Wx, Wy, Nx, Ny):
    mat = np.zeros((Nx,Ny))
    gray_scaled_mat = np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            mat[i][j] = phi(i/Nx,j/Ny, Wx, Wy)
            # gray scaling
            gray_scaled_mat[i, j] = (mat[i,j]+A)/(2*A)
    return mat, gray_scaled_mat

# present the signal phi aproximated by the matrix "mat"
def presentSignal(mat):
    imshow('Grayscaled Image', mat)
    waitKey()

# -------------- c --------------
# analytical results for comparison
analtical_range = 5000
analtical_HDE = 246740110 # HDE = Horizontal Derivative Energy
analtical_VDE = 3022566348 # VDE = Vertical Derivative Energy

def findNumericalRange(mat):
    phi_min = np.inf
    phi_max = -np.inf
    for row in range(len(mat)):
        for col in range(len(mat[row])):
            if mat[row][col] < phi_min :
                phi_min = mat[row][col]
            if mat[row][col] > phi_max :
                phi_max = mat[row][col]
    return phi_min, phi_max

# numericaly calculates and returns HDE, VDE
def findNumericalDE(mat):
    VDE = 0
    for row in range(len(mat)):
        for col in range(len(mat[row])-1):
            VDE += ((mat[row][col+1] - mat[row][col])**2)

    HDE = 0
    for row in range(len(mat)-1):
        for col in range(len(mat)):
            HDE += ((mat[row+1][col] - mat[row][col])**2)
    return HDE, VDE

def calcError(val1, val2):
    return (abs(val1-val2)/val2)*100

# -------------- d --------------

# caculate MSE given n_x, n_y, b
def calcMSE(x):
    mat, scaled = approxSignal(2, 7, 512, 512)
    HDE, VDE = findNumericalDE(mat)
    range = 5000
    term_1 = (1/(12*x[0]**2))*(HDE)
    term_2 = (1/(12*x[1]**2))*(VDE)
    term_3 = ((range**2)/(12*2**(2*x[2])))
    return term_1 + term_2 + term_3

# round according to input
def roundVals(n_x, n_y, b, decision_x, decision_y, decision_b):
    if decision_x == 0:
        new_x = math.floor(n_x)
    else:
        new_x = math.ceil(n_x)

    if decision_y == 0:
        new_y = math.floor(n_y)
    else:
        new_y = math.ceil(n_y)

    if decision_b == 0:
        new_b = math.floor(b)
    else:
        new_b = math.ceil(b)

    return new_x, new_y, new_b

# decide how to round based on a given B
def decideRound(n_x, n_y, b, B):
    curr_vals = roundVals(n_x, n_y, b, 0, 0, 0)
    curr_mse = calcMSE(curr_vals)
    for decision_x in range(2):
        for decision_y in range(2):
            for decision_b in range(2):
                tmp_vals = roundVals(n_x, n_y, b, decision_x, decision_y, decision_b)
                tmp_mse = calcMSE(tmp_vals)
                if tmp_mse<curr_mse and tmp_vals[0]*tmp_vals[1]*tmp_vals[2]<=B:
                    # we have a legal better allocation
                    curr_vals = tmp_vals
                    curr_mse = tmp_mse
    return curr_vals, curr_mse

# main function to minimize MSE with a given B
def optimizeMSE(B):
    initial_guess = np.array([B**(1/3), B**(1/3), B**(1/3)])
    bnds = ((1, None), (1, None), (1, None))
    cons = {'type': 'eq', 'fun': lambda x: (x[0] * x[1] * x[2]) - B}
    opt_alloc = minimize(calcMSE, initial_guess, bounds=bnds, constraints=cons)
    opt_n_x, opt_n_y, opt_b = opt_alloc.x
    tot_MSE = opt_alloc.fun

    print("results before rounding:")
    print("Nx =", opt_n_x, "\nNy =", opt_n_y, "\nb =", opt_b)
    print("Correponding MSE = ", tot_MSE)

    opt_ints, error_rnd = decideRound(opt_n_x, opt_n_y, opt_b, B)
    new_x, new_y, new_b = opt_ints

    print("results after rounding:")
    print("Nx =",new_x, "\nNy =", new_y, "\nb =", new_b)
    print("Correponding MSE = ", error_rnd)
    mat_e, gray_mat_e = approxSignal(2, 7, new_x, new_y)
    presentSignal(gray_mat_e)

# -------------- f --------------

# caculate MSE given n_x, n_y, b
def calcMSE_2(x, HDE, VDE, range):
    term_1 = (1/(12*x[0]**2))*(HDE)
    term_2 = (1/(12*x[1]**2))*(VDE)
    term_3 = ((range**2)/(12*2**(2*x[2])))
    return term_1 + term_2 + term_3

# searching procedure to find bit alloc given B
def searchAlloc(B, HDE, VDE, phi_range):
    curr_mse = calcMSE_2((1, 1, 1), HDE, VDE, phi_range)
    res = (1, 1, 1, curr_mse)
    # if it takes too long, decrease the ranges to 15, 700, 700
    # the results wont change :)
    for b in range(1, 25): 
        for nx in range(1, 1100):
            for ny in range(1, 1100):
                if b*nx*ny <= B:
                    tmp_mse = calcMSE_2((nx, ny, b), HDE, VDE, phi_range)
                    if tmp_mse < curr_mse:
                        curr_mse = tmp_mse
                        res = (nx, ny, b, curr_mse)
    return res


if __name__ == "__main__":
    # -------------- b --------------
    # approximating the signal with 512x512 grid
    mat, scaled = approxSignal(2, 7, 512, 512)
    presentSignal(scaled)

    # -------------- c --------------
    print()
    print("section c:")
    min_val, max_val = findNumericalRange(mat)
    print("numerical range is: [", max_val, ", ", min_val, "]. max_val-min_val=", max_val-min_val)
    HDE, VDE = findNumericalDE(mat)
    print("numerical HDE is:", HDE, ".", calcError(HDE, analtical_HDE), "%", "deviation from analytical")
    print("numerical VDE is:", VDE, ".", calcError(VDE, analtical_VDE), "%", "deviation from analytical")

    # -------------- e --------------
    # find optimal n_x, n_y, b given B using minimize func
    print()
    print("section e:")
    print("results for B_low:")
    optimizeMSE(5000) # B_low
    print("results for B_high:")
    optimizeMSE(50000) # B_high

    # -------------- g --------------
    # find optimal n_x, n_y, b given B using search procedure
    print()
    print("section g:")
    HDE, VDE = findNumericalDE(mat)
    phi_range = 5000
    print("results for B_low:")
    opt_nx, opt_ny, opt_b, tot_error = searchAlloc(5000, HDE, VDE, phi_range) # B_low
    print("nx =", opt_nx)
    print("ny =", opt_ny)
    print("b =", opt_b)
    print("Corresponding MSE= ", tot_error)

    mat_g, gray_mat_g = approxSignal(2, 7, opt_nx, opt_ny)

    imshow("searching procedure B_low", gray_mat_g)
    waitKey()

    print("results for B_high:")
    opt_nx, opt_ny, opt_b, tot_error = searchAlloc(50000, HDE, VDE, phi_range) # B_high
    print("nx =", opt_nx)
    print("ny =", opt_ny)
    print("b =", opt_b)
    print("Corresponding MSE= ", tot_error)

    mat_g, gray_mat_g = approxSignal(2, 7, opt_nx, opt_ny)
    imshow("searching procedure B_high", gray_mat_g)
    waitKey()

    # -------------- h --------------
    '''
    to avoid duplicating the entire __main__ code with the new vals,
    we changed the initial Wx&Wy for the output, and then returned the original vals
    all other functions beside main() are generic, so no additional change was needed.
    '''