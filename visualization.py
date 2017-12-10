import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def running_mean(arr, num):
    if len(arr) == 0:
        return [0]
    cumsum = np.cumsum(np.insert(arr, 0, [arr[0]]*num))
    return (cumsum[num:] - cumsum[:-num]) / num 

    
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

    
def show_losses_ex(ep, lr, tr_losses, va_losses, mean_win=30, log_scale=False):
    tr_rm = running_mean(tr_losses, mean_win)
    va_rm = running_mean(va_losses, mean_win)
    tr_loss = tr_rm[-1] if len(tr_rm)>0 else tr_losses[-1]
    va_loss = va_rm[-1] if len(va_rm)>0 else 0

    plt.title("ep:%.1f, lr:%.1e, valid:%.1e, train:%.1e" % (ep, lr, va_loss, tr_loss))
    plt.plot(tr_losses, 'c')
    va, = plt.plot(va_rm, 'r', label="valid")
    tr, = plt.plot(tr_rm, 'b', label="train")
    
    plt.grid(True)    
    if log_scale:
        plt.yscale("log")
    plt.legend(handles=[va, tr])

    clear_output(True)    
    plt.show()

   
    
def show_train_stats(ep, lr, tr_losses, va_losses, tr_accs, va_accs, neg_dist, pos_dist, mean_win=30, log_scale=False):
    plt.figure(figsize=(16,12))
    fontsize = 14

    # Loss
    plt.subplot(221)
    tr_means = running_mean(tr_losses, mean_win)
    va_means = running_mean(va_losses, mean_win)
    tr_loss = tr_means[-1]
    va_loss = va_means[-1]
    plt.title("Epoch %.2f | LR %.1e | Valid %.2e | Train %.2e" % (ep, lr, va_loss, tr_loss), fontsize=fontsize)
    if not log_scale:
        plt.plot(tr_losses, 'c')
        va, = plt.plot(va_means, 'r', label="Valid")
        tr, = plt.plot(tr_means, 'b', label="Train")
        plt.legend(handles=[va, tr])
    else:
        plt.yscale("log")
        tr, = plt.plot(tr_means, 'b', label="Train")
        plt.legend(handles=[tr])
    plt.grid(True)
   
    
    # Accuracy
    plt.subplot(222)
    tr_means = running_mean(tr_accs, mean_win)
    va_means = running_mean(va_accs, mean_win)
    tr_acc = tr_means[-1]
    va_acc = va_means[-1]
    plt.yscale('linear')
    plt.title('Accuracy | Valid %.2f%% | Train %.2f%%' % (va_acc*100, tr_acc*100), fontsize=fontsize)
    tr, = plt.plot(tr_means, 'b', label="Train")
    va, = plt.plot(va_means, 'r', label="Valid")
    plt.legend(handles=[tr, va], loc=0)
    plt.grid(True)
    
    # Distances
    plt.subplot(223)
    neg_means = running_mean(neg_dist, mean_win)
    pos_means = running_mean(pos_dist, mean_win)
    dif_means = [neg_means[i]-pos_means[i] for i in range(len(neg_means))]
    neg = neg_means[-1]
    pos = pos_means[-1]
    dif = dif_means[-1]
    plt.yscale('linear')
    plt.title('Distances on Train | Neg %.2f | Pos %.2f | Dif %.2f' % (neg, pos, dif), fontsize=fontsize)
    nd, = plt.plot(neg_means, 'r', label="Negative")
    pd, = plt.plot(pos_means, 'b', label="Positive")
    dd, = plt.plot(dif_means, 'g', label="Difference")
    plt.legend(handles=[nd, pd, dd], loc=0)
    plt.grid(True)

    # Show
    clear_output(True)    
    plt.show()