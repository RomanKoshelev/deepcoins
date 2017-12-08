import os

DatasetRootPath = "/netforge/datasets/private/roman/coins/2017-11-29/datasets/"

def get_dataset_path(path):
    assert path[0] != '/'
    return os.path.join(DatasetRootPath, path)
