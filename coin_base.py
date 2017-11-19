import numpy as np
from sklearn.neighbors import BallTree

class DataBase:
    def _get_embeds(self, images):
        outputs = self.model.run(images)
        if self.metric =='hamming':
            embeds  = np.sign((outputs+1)/2)
        elif self.metric =='euclidean':
            embeds  = outputs
        return embeds
        
    def build(self, model, images, metric):
        self.model   = model
        self.metric  = metric
        self.images  = np.copy(images)
        self.embeds  = self._get_embeds(images)
        self._tree   = BallTree(self.embeds, metric=metric)
        
    def query(self, images, k=1):
        embeds    = self._get_embeds(images)
        dist, ind = self._tree.query(embeds, k)
        return ind, dist
