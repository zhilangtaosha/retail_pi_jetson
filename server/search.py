import numpy as np
from utils import cosineDistance

def bruteforce(qfe, fdb, threshold):
    """
    glorious for loop through the face database
    """
    min_dst = 100
    best_match = None
    for doc in fdb:
        for feat in doc['feats']:
            feat_np = np.asarray(feat)
            cos_dst = cosineDistance(qfe, feat_np)
            if (cos_dst <= threshold) and (cos_dst < min_dst):
                min_dst = cos_dst
                best_match = doc
    return best_match
    
def unique_people_search(uuf, ruf, fdb, threshold):
    """
    TODO
    input: upload unique faces, refined unique faces, face database, cosine distance threshold
    output: unique identification with correct data
    """
    unique_people = None
    return unique_people
