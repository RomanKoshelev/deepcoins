import matplotlib.pyplot as plt
import itertools
import numpy as np
from timeit import default_timer as timer


def print_accuracy(dbase, request, k):
    def accuracy(ind, k):
        assert 0 < k <= ind.shape[1]
        err = 0
        N   = len(ind)
        for i in range(N):
            ok = False
            for j in range(k):
                if i==ind[i,j]:
                    ok = True
            err+= not ok
        return ((N-err)/N*100)

    start = timer()
    ind, dist = dbase.query(request, k)
    t = timer() - start
    n = request.shape[0]

    print("Database   : %s" % list(dbase.embeds.shape))
    print("Request    : %s" % list(request.shape))
    print("Performance: %.0f img/sec" % (n/t))
    print('-'*50)
    for k in range(1, ind.shape[1]+1):
        print("Accuracy@%d: %.1f%%" % (k, accuracy(ind, k)))
    print('-'*50)
    

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

    H,W = sheet.shape[:2]
    fw = 22
    fh = fw*H/W
    plt.figure(figsize = (fw,fh))
    plt.axis("off")
    plt.imshow(sheet, cmap='gray')
    plt.show()
    
    
def _show_similarity(img1, img2, sim, cols):
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

    
def plot_search_results(dbase, request, num, cols=4, k=0, sort=None, plot_limit=None):
    per = np.random.choice(range(len(request)), num, replace=False)
    pn  = plot_limit or len(request)
    request   = request[per]
    ethalons  = dbase.images
    ind, dist = dbase.query(request, k+1)
    response  = ethalons[ind[:,k]]
    d    = dist[:,k]
    dmin = np.min(d)
    dmax = np.max(d)
    dave = np.mean(d)
    print("Database   : %s" % list(dbase.embeds.shape))
    print("Request    : %s" % list(request.shape))    
    print("")
    print("Min distance: %.2f" % dmin)
    print("Max distance: %.2f" % dmax)
    print("Ave distance: %.2f" % dave)
    
    e   = 1e-8
    idx = range(len(d))
    sim = 1-(d-dmin+e)/(dmax-dmin+e*10)
    sim = np.minimum(sim, 1.)
    sim = np.maximum(sim, 0.)
    if sort=='asc':
        idx = sorted(idx, key=lambda k: d[k])
    elif sort=='desc':
        idx = sorted(idx, key=lambda k: 1-d[k])
    request  = request [idx][:pn]
    response = response[idx][:pn]
    sim      = sim     [idx][:pn]
    
    _show_similarity(request, response, sim, cols)