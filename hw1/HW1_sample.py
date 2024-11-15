import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def MSE(img, q_img):
    # flat_img  = img.flatten()
    # flat_q_img = q_img.flatten()
    # err = np.sum(np.power((flat_img - flat_q_img), 2))
    # err /= (img.shape[0] * img.shape[1])
    # return err
    width, height = img.shape
    size = width*height
    error = 0
    for x in range(width):
        for y in range(height):
            error += np.power(img[x][y] - q_img[x][y] , 2)
    return error / size


def MAD(img, q_img):
    flat_img  = img.flatten()
    flat_q_img = q_img.flatten()
    err = np.sum(np.abs(flat_img - flat_q_img))
    err /= (img.shape[0] * img.shape[1])
    return err 


def subsample_img(img, D):
    new_size  = img.shape / np.power(D,1)
    new_img = np.zeros((int(new_size[0]),int(new_size[1])))
    for i in range(0,img.shape[0],D):
        for j in range(0,img.shape[1],D):
            vec = []
            for iner_i in range(D):
                for iner_j in range(D):
                    vec.append(img[iner_i+i][iner_j+j])
            new_val =  np.average(vec)
            new_img[int(i/D), int(j/D)] = new_val
    return new_img


# def reconstruct_img(img, D):
#     new_size  = [img.shape[0] * D, img.shape[1] * D]
#     new_img = np.zeros((int(round(new_size[0])),int(round(new_size[1]))))
#     for i in range(0,img.shape[0]):
#         for j in range(0,img.shape[1]):
#             val = img[i,j]
#             for iner_i in range(D):
#                 for iner_j in range(D):
#                     new_img[int(i*D + iner_j), int(j*D + iner_j)] = val
#     return new_img

def reconstruct_img(img, D):
    
    new_size = [img.shape[0] * D, img.shape[1] * D]
    new_img = np.zeros((int(new_size[0]), int(new_size[1])))
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            new_img[i, j] = img[int(i / D), int(j / D)]
    return new_img

if __name__ == "__main__":
    img = np.array(Image.open('rino.jpg')).astype(np.int64)
    mse_err = []
    fig = plt.figure(figsize=(8, 4))
    rows = 2
    columns = 4
    for i in range(1,9):
        D = 2**i
        sampled_img = subsample_img(img, D)
        rec_img = reconstruct_img(sampled_img, D)
        mse_err.append(MSE(img, rec_img))
        fig.add_subplot(rows, columns, i)
        fig.set_label('MSE Error')
        plt.xlabel('D : {}'.format(D))
        plt.imshow(sampled_img , cmap='gray', vmin=0, vmax=255)
    fig.suptitle('MSE Subsampling')
    plt.show()
    plt.cla()
    
    plt.plot(mse_err)
    plt.xlabel('MSE - D')
    plt.ylabel('Error')
    plt.xticks(np.arange(8), np.arange(1, 9))
    plt.show()

    mad_err = []
    fig = plt.figure(figsize=(8, 4))
    rows = 2
    columns = 4
    for i in range(1,9):
        D = 2**i
        sampled_img = subsample_img(img, D)
        rec_img = reconstruct_img(sampled_img, D)
        mad_err.append(MAD(img, rec_img))
        fig.add_subplot(rows, columns, i)
        fig.set_label('MSE Error')
        plt.xlabel('D : {}'.format(D))
        plt.imshow(sampled_img , cmap='gray', vmin=0, vmax=255)
    fig.suptitle('MAD Subsampling')
    plt.show()
    plt.cla()
    
    plt.plot(mad_err)
    plt.xlabel('MAD - D')
    plt.ylabel('Error')
    plt.xticks(np.arange(8), np.arange(1, 9))
    plt.show()


    fig = plt.figure(figsize=(8, 4))
    rows = 2
    columns = 4
    for i in range(1, 9):
        D = np.power(2, i)
        sampled_img = subsample_img(img, D)
        rec_img = reconstruct_img(sampled_img, D)
        err = MSE(img, rec_img)
        fig.add_subplot(rows, columns, i)
        fig.set_label('MSE Error')
        plt.xlabel('D : {}'.format(D))
        plt.imshow(rec_img, cmap='gray', vmin=0, vmax=255)
    fig.suptitle('MSE Reconstruction')
    plt.show()
    plt.cla()


    fig = plt.figure(figsize=(8, 4))
    for i in range(1, 9):
        D = np.power(2, i)
        sampled_img = subsample_img(img, D)
        rec_img = reconstruct_img(sampled_img, D)
        err = MAD(img, rec_img)
        fig.add_subplot(rows, columns, i)
        fig.set_label('MAD Error')
        plt.xlabel('D : {}'.format(D))
        plt.imshow(rec_img, cmap='gray', vmin=0, vmax=255)
    fig.suptitle('MAD Reconstruction')
    plt.show()
    
