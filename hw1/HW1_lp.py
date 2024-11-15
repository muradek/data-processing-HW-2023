from re import I
from struct import pack
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

LIMIT = 100000
EPSILON = 1/0.000000000003
def calculate_error(f_i, f_i_hat, w_matrix, p):
    total_error = 0
    for i in range(int(f_i.shape[0])):
            for j in range(int(f_i.shape[1])):
                error = w_matrix[i][j] * np.power(np.abs(f_i[i][j] - f_i_hat[i][j]), p) 
                total_error += error
    return total_error

def I_W_LS(f_i, N, p, delta):
    #initilize a weights matrix to 1
    global LIMIT
    global EPSILON
    S = int(512/N)
    weights_mat = [[1]*(int(f_i.shape[0])) for i in range(int(f_i.shape[1]))]
    f_i_hat = np.copy(f_i)
    prev_error = 0
    error = np.inf
    iteration = 0
    while True:
        #go over each square in grid
        for i in range(0, int(f_i_hat.shape[0]), S):
            for j in range(0, int(f_i_hat.shape[1]), S):
                val = 0
                sum_weights = 0
                #sample each grid, and generate an optimal sample
                for inner_i in range(i, i + S):
                    for inner_j in range(j, j + S):
                        val += (f_i_hat[inner_i][inner_j])*weights_mat[inner_i][inner_j]
                        sum_weights += weights_mat[inner_i][inner_j]
                f_i_next = round(val / sum_weights)
                #calculate the new const aprox funtion, update the grid
                #update the weights at the same time
                for inner_i in range(i, i + S):
                    for inner_j in range(j, j + S):
                        f_i_hat[inner_i][inner_j] = f_i_next
                        w_i_next = np.minimum(weights_mat[inner_i][inner_j] * 
                        np.power((np.abs(f_i[inner_i][inner_j] - f_i_hat[inner_i][inner_j])), p-2) 
                        , EPSILON)
                        if w_i_next ==0:
                            w_i_next = EPSILON
                        weights_mat[inner_i][inner_j] = w_i_next
        prev_error = error
        error = calculate_error(f_i, f_i_hat, weights_mat, p)
        diff = prev_error - error
        if prev_error - error < delta:
            return f_i_hat, round(error)
        iteration += 1
        if iteration > LIMIT:
            raise ValueError("ran one million iterations")


def L_1(f_i, N):
    #initilize a weights matrix to 1
    global LIMIT
    global EPSILON
    S = int(512/N)
    weights_mat = [[1]*(int(f_i.shape[0])) for i in range(int(f_i.shape[1]))]
    f_i_hat = np.copy(f_i)
    #go over each square in grid
    for i in range(0, int(f_i.shape[0]), S):
        for j in range(0, int(f_i.shape[1]), S):
            vals = []
            #sample each grid, and generate an optimal sample
            for inner_i in range(i, i + S):
                for inner_j in range(j, j + S):
                    vals.append(f_i[inner_i][inner_j])
            opt_val = round(np.median(vals))
            for inner_i in range(i, i + S):
                for inner_j in range(j, j + S):
                    f_i_hat[inner_i][inner_j] = opt_val
    error = calculate_error(f_i, f_i_hat, weights_mat, 1)
    return f_i_hat, error

if __name__ == '__main__':

    fig = plt.figure(figsize=(8, 4))
    rows = 1
    columns = 2
    img = np.array(Image.open('rino.jpg')).astype(np.int64)
    N = 128
    P = 1.0
    delta = 10000000000
    new_img, I_err = I_W_LS(img, N, P, delta)
    fig.add_subplot(rows, columns, 1)
    plt.xlabel(f'L_1 approximate algorithm, delta : {delta}, error : {I_err}')
    plt.imshow(new_img
                , cmap='gray', vmin=0, vmax=255)
    # plt.show()
    # plt.cla()
    
#exercize 3.b
    
    new_img, L_err = L_1(img, N)
    fig.add_subplot(rows, columns, 2)
    plt.xlabel(f'L_1 exact algorithm, error : {L_err}')
    plt.imshow(new_img
                , cmap='gray', vmin=0, vmax=255)
    fig.suptitle(f'Uniform sampling on a square grid of size: {N}X{N}')
    plt.show()
    plt.cla()


    N = 128
    P = 1.5
    delta = 100
    new_img, I_err = I_W_LS(img, N, P, delta)
    plt.xlabel(f'L_{P} approximate algorithm, delta : {delta}, error : {I_err}, N : {N}')
    plt.imshow(new_img
                , cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.cla()

    N = 256
    P = 1.5
    delta = 100
    new_img, I_err = I_W_LS(img, N, P, delta)
    plt.xlabel(f'L_{P} approximate algorithm, delta : {delta}, error : {I_err}, N : {N}')
    plt.imshow(new_img
                , cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.cla()

    N = 128
    P = 4
    delta = 100
    new_img, I_err = I_W_LS(img, N, P, delta)
    plt.xlabel(f'L_{P} approximate algorithm, delta : {delta}, error : {I_err}, N : {N}')
    plt.imshow(new_img
                , cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.cla()

    N = 256
    P = 4
    delta = 100
    new_img, I_err = I_W_LS(img, N, P, delta)
    plt.xlabel(f'L_{P} approximate algorithm, delta : {delta}, error : {I_err}, N : {N}')
    plt.imshow(new_img
                , cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.cla()