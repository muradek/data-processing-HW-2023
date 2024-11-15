from statistics import mean
from tkinter.tix import REAL
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import scipy.linalg

# set given values 
N = 64
c = 0.6
REALIZATIONS_QUANTITY = 100000
M_mean = 0
L_std = np.sqrt((N / 2) * (1 - c))
L_mean = 0
M_std = np.sqrt(c)

# ------ section a ------

# generate many realizationf of the given class
def createRandomRealizations():
    realiztion_mat = []
    for j in range(REALIZATIONS_QUANTITY):
        K = np.random.randint(0, (N // 2))
        L = np.random.normal(L_mean, L_std)
        M = np.random.normal(M_mean, M_std)
        vec = np.zeros(N)
        for i in range(N):
            if i == K or i == K + N//2:
                vec[i] = M + L
            else:
                vec[i] = M

        realiztion_mat.append(vec)
    return np.asmatrix(realiztion_mat)

# compute the empirical mean of the vectors
def computeMean(mat):
    mean_vec = np.zeros(N)
    for i in range(len(mat)):
        mean_vec = np.add(mean_vec, mat[i])
    return mean_vec / (len(mat)+1)
    
# compuute autocorrelation matrix
def computeAutocorrelationMatrix(mat):
    R_phi = np.zeros((N, N))
    for i in range(len(mat)):
        vec = mat[i]
        R_phi += np.dot(vec.T, vec)
    return (R_phi / len(mat))

# plot the mean & autocorrelation matrix
def plotSectionA(mean, R_phi):
    fig, axs = plt.subplots(1, 2)

    # R_phi
    axs[0].set_title('Autocorrelation Matrix')
    axs[0].imshow(R_phi)

    # Mean Vector
    axs[1].plot([i for i in range(N)], mean.T)
    axs[1].set_xlabel('vector entry')
    axs[1].set_ylabel('value')
    axs[1].set_title('Mean Vector')

    plt.show()


# ------ section b ------

# numerically construct the Wiener filter
def constructWiener(H_op, R_phi, n_var):
    inner_res = np.linalg.pinv(H_op.dot(R_phi).dot(H_op.T) + n_var * np.identity(N))
    outer_res = R_phi.dot(H_op.T)
    final_res = outer_res.dot(inner_res)
    return final_res

# plot the constructed Wienre matrix
def plotWiener(filter):
    fig, axes = plt.subplots()
    axes.imshow(filter)
    axes.set_title('Weiner Filter Matrix')
    plt.show()

# construct the noise class realizations
def constructNoise(n_var, quantity):
    return np.random.normal(0, n_var, (quantity, N))

def calcMse(old_vec, new_vec):
    sum = np.sum(np.power(old_vec - new_vec, 2))
    avg = sum / (old_vec.shape[0] * new_vec.shape[1])
    return avg

# plot the clean, noised and denoised realiztions 
def plotRealizations(clean, noised, denoised):
    fig, axs = plt.subplots(3, 1)
    mse = round(calcMse(clean, denoised), 2)
    to_print = {'clean': clean, 'noised': noised, 'denoised, MSE : {}'.format(mse): denoised}
    colors = ['b-', 'g-', 'r-']
    # Mean Vector
    index = 0
    for label, vec in to_print.items():
        axs[index].plot([i for i in range(N)], vec, colors[index])
        axs[index].set_xlabel('Index')
        axs[index].set_ylabel('Value')
        axs[index].set_title(label)
        index += 1
    plt.tight_layout()
    plt.show()


# ------ sections c+d ------

def funcH():
    start = [-5 / 2, 4 / 3, -1 / 12]
    end = [-1 / 12, 4 / 3]
    middle_length = N - len(start) - len(end)
    h_1 = deque(start + [0 for i in range(middle_length)] + end)
    return scipy.linalg.circulant(h_1)

# ------ section e ------

def sectionE():
    H = funcH()
    inv_H = np.linalg.pinv(H)
    vec = np.array([1 for val in range(N)])
    fig, axs = plt.subplots(1, 2)
    print(inv_H.dot(vec))
    axs[0].imshow(H.dot(inv_H))
    axs[0].set_title('H * H_inv')

    axs[1].imshow(inv_H.dot(H))
    axs[1].set_title('H_inv * H')
    phi_1 = np.zeros((N, 1))
    phi_2 = np.zeros((N, 1))
    phi_1[:, 0] = np.array([1 for val in range(N)])
    phi_2[:, 0] = np.array([100 for val in range(N)])
    fig, axs = plt.subplots(3, 2)
    axs[0][0].plot(phi_1)
    axs[0][0].set_xlabel('Index')
    axs[0][0].set_ylabel('Value')
    axs[0][0].set_title('Phi_1')
    axs[0][1].plot(phi_2)
    axs[0][1].set_xlabel('Index')
    axs[0][1].set_ylabel('Value')
    axs[0][1].set_title('Phi_2')

    axs[1][0].plot(phi_1 - phi_2)
    axs[1][0].set_xlabel('Index')
    axs[1][0].set_ylabel('Value')
    axs[1][0].set_title('Phi_1 - Phi_2')

    axs[1][1].plot(inv_H.dot(phi_1))
    axs[1][1].set_xlabel('Index')
    axs[1][1].set_ylabel('Value')
    axs[1][1].set_title('H_inv*phi_1')

    axs[2][1].plot(inv_H.dot(phi_2))
    axs[2][1].set_xlabel('Index')
    axs[2][1].set_ylabel('Value')
    axs[2][1].set_title('H_inv*phi_2')

    axs[2][0].plot(inv_H.dot(phi_1) - inv_H.dot(phi_2))
    axs[2][0].set_xlabel('Index')
    axs[2][0].set_ylabel('Value')
    axs[2][0].set_title('H_inv * phi_1 - H_inv * phi_2')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # ------ section a ------
    realizations = createRandomRealizations()
    mean = computeMean(realizations)
    R_phi = computeAutocorrelationMatrix(realizations)
    plotSectionA(mean, R_phi)

    # ------ section b ------
    real_t = realizations.T
    H = np.identity(N)
    n_var = 1
    filter = constructWiener(H, R_phi, n_var)
    plotWiener(filter)
    noise = constructNoise(n_var, REALIZATIONS_QUANTITY).T
    noised_realizations = H.dot(real_t) + noise
    denoised_realizations = filter.dot(noised_realizations)
    for i in range(3):
        plotRealizations(real_t[:, i], noised_realizations[:, i], denoised_realizations[:, i])
    print(calcMse(real_t, denoised_realizations))

    # ------ section c ------
    real_t = realizations.T
    H = funcH()
    n_var = 1
    filter = constructWiener(H, R_phi, n_var)
    plotWiener(filter)
    noise = constructNoise(n_var, REALIZATIONS_QUANTITY).T
    noised_realizations = H.dot(real_t) + noise
    denoised_realizations = filter.dot(noised_realizations)
    for i in range(3):
        plotRealizations(real_t[:, i], noised_realizations[:, i], denoised_realizations[:, i])
    print(calcMse(real_t, denoised_realizations))
    
    # ------ section d ------
    real_t = realizations.T
    H = funcH()
    n_var = 5
    filter = constructWiener(H, R_phi, n_var)
    plotWiener(filter)
    noise = constructNoise(n_var, REALIZATIONS_QUANTITY).T
    noised_realizations = H.dot(real_t) + noise
    denoised_realizations = filter.dot(noised_realizations)
    for i in range(3):
        plotRealizations(real_t[:, i], noised_realizations[:, i], denoised_realizations[:, i])
    print(calcMse(real_t, denoised_realizations))

    # ------ section e ------
    sectionE()

