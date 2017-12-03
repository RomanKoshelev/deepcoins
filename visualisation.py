import numpy as np
import matplotlib.pyplot as plt
import itertools
from IPython.display import clear_output

def _show_images(images, image_shape, rows, cols):
    h = image_shape[0]
    w = image_shape[1]

    images = np.reshape(images, [-1, h*w])

    n = rows*cols
    img = images[:n]
    img = np.reshape(img, [n*h, w])
    sheet = np.zeros([rows*h, cols*w])

    for i, j in itertools.product(range(rows), range(cols)):
        H = (i*cols+j) * h
        sheet[i*h:i*h+h, j*w:j*w+w] = img[H: H+h, 0:w]

    plt.figure(figsize = (18,8))
    plt.axis("off")
    plt.imshow(sheet, cmap='gray')
    plt.show()
    
    
def show_losses(losses, step, step_num, mean_win=10):
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 

    if mean_win>0:
        plt.plot(losses, 'c')
        plt.plot(running_mean(losses, mean_win), 'b')
        loss = np.mean(losses[-mean_win:])
    else:
        plt.plot(losses)
        loss = losses[-1]

    plt.grid(True)
    plt.title("step: %d/%d [%.0f%%], loss: %.2e" % (step+1, step_num, 100*(step+1)/step_num, loss))
    clear_output(True)
    plt.show()

    
def show_losses_ex(step, tr_losses, va_losses, mean_win=30, log_scale=False):
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 

    tr_rm = running_mean(tr_losses, mean_win)
    va_rm = running_mean(va_losses, mean_win)
    tr_loss = tr_rm[-1] if len(tr_rm)>0 else tr_losses[-1]
    va_loss = va_rm[-1] if len(va_rm)>0 else 0

    plt.title("step: %d, va_loss: %.4f, tr_loss: %.4f" % (step+1, va_loss, tr_loss))
    plt.plot(tr_losses, 'c')
    va, = plt.plot(va_rm, 'r', label="valid")
    tr, = plt.plot(tr_rm, 'b', label="train")
    
    plt.grid(True)    
    if log_scale:
        plt.yscale("log")
    plt.legend(handles=[va, tr])

    clear_output(True)    
    plt.show()

    
def show_similarity(img1, img2, sim, cols=4):
    num = img1.shape[0]
    h   = img1.shape[1]
    w   = img1.shape[2]
    w3  = 32
    img1 = np.copy(img1.reshape([num*h,w]))
    img2 = np.copy(img2.reshape([num*h,w]))
    img3 = np.ones([num*h, w3])
    for i in range(num):
        s = sim[i]
        assert(0.<=s<=1.)
        img3[i*h:i*h+h] = s
    img1[:,0] = 0
    img3[:,w3-1] = 0
    img3[:,0] = 0
    sheet = np.concatenate([img1, img2, img3], axis=1)
    sheet[np.arange(0,num*h,h)-1,:] = 0
    sheet[np.arange(0,num*h,h)+1,:] = 0
    sheet = np.minimum(sheet, 1)
    _show_images(images=sheet, image_shape=[h, 2*w+w3], cols=cols, rows=num//cols)
