from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy as np
filename = "rino.jpg"

def histogram():
    global filename
    total  = 0
    with Image.open(filename) as image:
        width, height = image.size
        histogram = [0] * 256
        for x in range(width):
            for y in range(height):
                value = image.getpixel( (x, y) )
                histogram[value] += 1
                total += 1
    plt.plot(histogram)
    plt.show()

# excersize 1
# histogram()  

def get_pdf():
    global filename
    total  = 0
    with Image.open(filename) as image:
        width, height = image.size
        pdf = [0] * 256
        for x in range(width):
            for y in range(height):
                value = image.getpixel( (x, y) )
                pdf[value] += 1
                total += 1
    for i in range(len(pdf)):
         pdf[i] = pdf[i]/total
    return pdf 
    # scales = list(range(256))           
    # plt.plot(scales, pdf)
    # plt.xlabel('Gray scale')
    # plt.ylabel('counts')
    # plt.show()


#quantization function of x based on k uniform intervals
def Q_k_x(k, x):
    delta = 256/k
    return ((math.floor(x/delta)+0.5) * delta)

#MSE with K representation levels
def MSE_K(k):
    global filename
    total_err = 0
    size = 0
    with Image.open(filename) as image:
        width, height = image.size
        size = width*height
        for x in range(width):
            for y in range(height):
                value = image.getpixel( (x, y) )
                total_err += ((value - Q_k_x(k, value))**2)
    return (total_err / size)


def plot_MSE_by_bit():
    err_by_b = []
    b_list = []
    for b in range(1,9):
        err_by_b.append(MSE_K(2**b))
        b_list.append(b)
    plt.plot(b_list , err_by_b)
    plt.xlabel('bits')
    plt.ylabel('MSE')
    plt.show()

#excersize 2.a
# plot_MSE_by_bit()



#compute initial decision levels and representation levels with b bits system
def ret_dec_rep_levels_by_bit(b):
    rep_levels = []
    dec_levels = []
    k = 2**b
    delta = 256/k
    for i in range(1,k+1):
        rep = (i - 0.5) * delta
        rep_levels.append(rep)
    for i in range(1,k):
        dec_levels.append(delta*i)
    return dec_levels, rep_levels

#plot the uniform deicision and rep levels
def plot_uni_dec_rep():
    x = np.array(range(257))
    y = [0]*257
    for i in range(1,9):
        dec_levs, rep_levs =  ret_dec_rep_levels_by_bit(i)
        # for i in dec_levs:
        #     y[int(round(i))] +=1
        # dec_levs.append(256)
        # dec_levs.insert(0,0)
        plt.plot(dec_levs, 'bo')
        if i <= 4:
            plt.xticks(np.arange(2 ** i - 1), np.arange(1, 2 ** i ))
        plt.yticks(np.arange(0, 275, 25), np.arange(0, 275, 25))
        plt.xlabel('Decision Levels')
        plt.ylabel('Grayscale')
        plt.show()

        plt.cla()
        if i <= 4:
            plt.xticks(np.arange(2 ** i - 1), np.arange(1, 2 ** i ))
        plt.xlabel('Representation Levels')
        plt.ylabel('Grayscale')
        plt.plot(rep_levs, 'bo')
        plt.yticks(np.arange(0, 275, 25), np.arange(0, 275, 25))
        plt.show()
        plt.cla()
        


#exercise 2.b
# plot_uni_dec_rep()

def rep_i(pdf, start, stop):
    total_p = 0
    total_px = 0
    start = int(start)
    stop = int(stop)
    for i in range(start, stop):
        total_p += pdf[i]
        total_px += i*pdf[i]
    return (total_px / total_p) if total_p else (start+stop) / 2

#given k+1 decision levels(starting at 0 and ending at 256) returns k representation levels 
def comp_rep_levels(pdf, dec_levs):
    rep_levels = []
    d_i_minus_1 = 0
    for i in range(len(dec_levs)):
        d_i = dec_levs[i]
        rep_levels.append(rep_i(pdf, d_i_minus_1, d_i))
        d_i_minus_1 = d_i
    rep_levels.append(rep_i(pdf, d_i_minus_1, 256))
    return rep_levels

def comp_dec_levels(rep_levels):
    dec_levels = []
    for i in range(len(rep_levels)-1):
        dec_level = (round((rep_levels[i] + rep_levels[i+1]) / 2.0))
        dec_levels.append(dec_level)
    return dec_levels

#get the quantization of a given x, given rep and dec levels, a better implementation would be bin_search
def q_x_r(dec_levels, rep_levels, x):
    if 0 <= x < dec_levels[0]:
            return rep_levels[0]
    if dec_levels[len(dec_levels)-1] <= x <=256:
        return rep_levels[len(rep_levels)-1]
    for i in range(1,len(dec_levels)):
        if dec_levels[i-1] <= x <= dec_levels[i]:
            return rep_levels[i]

#MSE with given decision and representation levels
def MSE_d_r(dec_levs, rep_levs, pdf):
    total_err = 0
    for i in range(len(dec_levs)):
        bottom = int(dec_levs[i-1] if i else 0)
        top =  int(dec_levs[i])
        for j in range(bottom, top+1):
            total_err += (j - rep_levs[i]) ** 2 * pdf[j]
            #i though it should be (j - rep_levs[i])**2, and then divided at end 
    bottom = int(dec_levs[-1])
    top = 256
    for j in range(bottom, top):
        total_err += (j - rep_levs[-1]) ** 2 * pdf[j]
    return (total_err) # ithought you should divide by total size


#MSE with given decision and representation levels of an image
def MSE_I_d_r(dec_levs, rep_levs):
    global filename
    total_err = 0
    size = 0
    with Image.open(filename) as image:
        width, height = image.size
        size = width*height
        for x in range(width):
            for y in range(height):
                value = image.getpixel( (x, y) )
                q_val = q_x_r(dec_levs, rep_levs, value)
                total_err += ((value - q_val)**2)
    return (total_err / size)


#max lloyd algorith, recieves a pdf, initial  decision levels and e, the convergence stopping  condition
def max_Lloyd(pdf, dec_levs, e):
    first_rep = rep_levels = comp_rep_levels(pdf, dec_levs)
    first_dec = dec_levs = comp_dec_levels(rep_levels)
    cur_mse = MSE_d_r(dec_levs, rep_levels, pdf)
    rep_levels = comp_rep_levels(pdf, dec_levs)
    dec_levs = comp_dec_levels(rep_levels)
    mse = MSE_d_r(dec_levs, rep_levels, pdf)
    if cur_mse - mse >= e:
        return cur_mse, first_dec, first_rep
    while (cur_mse - mse) >= e:
        dec_levs = comp_dec_levels(rep_levels)
        rep_levels = comp_rep_levels(pdf, dec_levs)
        cur_mse = mse
        mse = MSE_d_r(dec_levs, rep_levels, pdf)
    return cur_mse, dec_levs, rep_levels


 #plots the MSE according to bit representation
def ML_MSE_by_bit():
    mse_by_bit = []
    pdf = get_pdf()
    for i in range(1,9):
        dec_levels, r_l =  ret_dec_rep_levels_by_bit(i)
        mse, final_dec_levs, final_rep_levs = max_Lloyd(pdf, dec_levels, 10)
        mse_by_bit.append(mse)
    bits = list(range(1,9))           
    plt.plot(bits, mse_by_bit)
    plt.xlabel('bits')
    plt.ylabel('MSE by bit')
    plt.show()



#plot the final decision levels and representation levels by bit
def ML_MSE_by_bit(b):
    pdf = get_pdf()
    for i in range(1,b+1):
        dec_levs, rep_levs =  ret_dec_rep_levels_by_bit(i)
        mse, final_dec_levs, final_rep_levs = max_Lloyd(pdf, dec_levs, 10)
        plt.plot(final_dec_levs, 'bo')
        if i <= 4:
            plt.xticks(np.arange(2 ** i - 1), np.arange(1, 2 ** i ))
        plt.yticks(np.arange(0, 275, 25), np.arange(0, 275, 25))
        plt.xlabel('Decision Levels')
        plt.ylabel('Grayscale')
        plt.show()

        plt.cla()
        if i <= 4:
            plt.xticks(np.arange(2 ** i - 1), np.arange(1, 2 ** i ))
        plt.xlabel('Representation Levels')
        plt.ylabel('Grayscale')
        plt.plot(final_rep_levs, 'bo')
        plt.yticks(np.arange(0, 275, 25), np.arange(0, 275, 25))
        plt.show()

# excersize 4b
ML_MSE_by_bit(8)
#opened questions: the mse in the uniform quantizer

if __name__ == '__main__':
    # excersize 1
    histogram()
    #excersize 2.a
    plot_MSE_by_bit()
    #exercise 2.b
    plot_uni_dec_rep()
    # excersie 4.a
    ML_MSE_by_bit()
    # excersize 4b
    ML_MSE_by_bit(8)