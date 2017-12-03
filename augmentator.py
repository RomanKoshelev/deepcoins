import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

class Augmentator:
    def __init__(self):
        ia.seed(None)
        do = iaa.Sometimes
        self._seq = iaa.Sequential([
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
            ])
    def augment(self, images):
        return self._seq.augment_images(images)