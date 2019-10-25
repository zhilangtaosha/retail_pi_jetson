import numpy as np
from sklearn.cluster import DBSCAN

def face_clustering(feat_queue):
    """
    get clustered faces based on embedded feature
    """
    if len(feat_queue) == 1:
        return np.asarray([0])
    print(len(feat_queue), feat_queue[0].shape)
    np_feats = np.asarray(feat_queue)
    np_feats = np.squeeze(np_feats)
    print(np_feats.shape)
    cluster = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(np_feats)
    return cluster.labels_
