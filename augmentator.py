import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

do = iaa.Sometimes

class Augmentator:
    
    Sequences = {
        # ============================================        
        'aug1': iaa.Sequential([
            iaa.Grayscale(alpha=(1.0, 1.0)),
            do(1., iaa.Affine(
                scale             = {"x": (+0.8, +1.0), "y": (+0.8, +1.0)},
                translate_percent = {"x": (-0.1, +0.1), "y": (-0.1, +0.1)},
                rotate            = (-45, +45),
                shear             = ( -5,  +5),
                order             = [0, 1],
                cval              = (220, 255),
            )),
            do(.5, iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))),
            do(.5, iaa.Multiply((0.5, 1.2))),
            do(.3, iaa.GaussianBlur(sigma=(0, 1.0))),
            do(.5, iaa.ContrastNormalization((0.5, 1.5))),
            do(.5, iaa.FrequencyNoiseAlpha(
                exponent       = (-4, -1),
                first          = iaa.Multiply(iaa.Choice((0.5, 2.0)), per_channel=False),
                size_px_max    = 32,
                upscale_method = "linear",
                iterations     = 1,
                sigmoid        = True)),
            ]),
        # ============================================

        
        # ============================================
        'aug2': iaa.Sequential([
            iaa.Grayscale(alpha=(1.0, 1.0)),
            do(1., iaa.Affine(
                scale             = {"x": (+0.95, +1.0), "y": (+0.95, +1.0)},
                translate_percent = {"x": (-0.05, +0.05), "y": (-0.05, +0.05)},
                rotate            = (-15, +15),
                shear             = ( -1,  +1),
                order             = [0, 1],
                cval              = (230, 255),
            )),
            do(.5, iaa.GaussianBlur(sigma=(.0,.5))),
            do(.5, iaa.AdditiveGaussianNoise(scale=(0, 10))),
            
            do(1., iaa.Add((-30,+30))),
            do(1., iaa.Multiply((.95, 1.05))),
            do(1., iaa.ContrastNormalization((0.75, 1.25))),
            
            do(.2, iaa.FrequencyNoiseAlpha(
                exponent       = -4,
                first          = iaa.Invert(1, per_channel=False),
                size_px_max    = 64,
                upscale_method = "linear",
                iterations     = 1,
                sigmoid        = True)),
            do(.5, iaa.FrequencyNoiseAlpha(
                exponent       = (-5, -2),
                first          = iaa.Multiply(iaa.Choice((0.7, 1.1)), per_channel=False),
                size_px_max    = 32,
                upscale_method = "linear",
                iterations     = 1,
                sigmoid        = True)),
            ]),        
        # ============================================
    }
    
    def __init__(self, name, seed=None):
        ia.seed(seed)
        self.name = name
        self._seq = self.Sequences[name]
        
    def augment(self, images):
        return self._seq.augment_images(images)