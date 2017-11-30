import numpy as np
import cv2
from scipy.ndimage.interpolation import rotate

class Augmentator:
    _params = {
        'smooth'    : .8,
        'angle'     : 45,
        'brightness': .5,
        'inverse'   : .5,
        'contrast'  : [.25, 2.5],
    }
    
    def __init__(self, cache_size=100, rate=1):
        self._cache      = dict()
        self._rate       = rate        
        self._cache_size = cache_size
        
    def _augment(self, im):
        rate  = self._rate
        shape = im.shape
        w = shape[1]
        h = shape[0]
        
        # cache
        hc = hash(im.tostring())
        cached = self._cache.get(hc, [])
        if len(cached) >= self._cache_size:
            return cached[np.random.randint(self._cache_size)]
        
        # smooth
        p = self._params['smooth']
        k = p*np.random.random(1)**2 * rate
        if k > 0.1:
            kernel = np.ones([5,5],'float32')/25
            old= np.copy(im)
            im = cv2.filter2D(im,-1,kernel)
            im = np.reshape(im, shape)
            im = k*im + (1-k)*old
            del old

        # angle
        p = self._params['angle']
        k = np.random.uniform(-p, p) * rate
        b = np.mean(im[:10,:10])
        im[0,:]   = b
        im[h-1,:] = b
        im[:,0]   = b
        im[:,w-1] = b
        im = rotate(im, k, reshape=False, mode='nearest')
        
        # brightness
        p = self._params['brightness'] * rate
        k = np.random.uniform(1-p, 1+p)
        im = im * k
        im = np.minimum(im,1)
        im = np.maximum(im,0)

        # contrast
        p = self._params['contrast']
        k = rate*np.random.uniform(*p) + (1-rate)
        im = np.power(im, k)
            
        # inverse
        p = self._params['inverse'] * rate
        k = np.random.random(1)
        if k < p:
            im = 1-im        
            
        # bounding
        assert np.all(im<=1)
        assert np.all(im>=0)

        # cache
        cached.append(im)
        self._cache[hc] = cached
        return im.astype('float16')
    
    def augment(self, images):
        images = np.copy(images)
        if self._rate > 0:
            for i in range(len(images)):
                images[i] = self._augment(images[i])
        return images
    