import numpy as np
from sklearn.neighbors import BallTree

class DataBase:
    def build(self, model, images):
        self.model   = model
        self.images  = np.copy(images)
        self.embeds  = self.model.forward(images)
        self._tree   = BallTree(self.embeds, metric='euclidean')
        
    def query(self, images, k=1):
        embeds    = self.model.forward(images)
        dist, ind = self._tree.query(embeds, k)
        return ind, dist
