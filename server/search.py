import numpy as np
from utils import cosineDistance, merge_data

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
            # cos_dst = cosineDistance(qfe, feat)
            if (cos_dst <= threshold) and (cos_dst < min_dst):
                min_dst = cos_dst
                best_match = doc['_id']
            print("cos dst: ", cos_dst, doc["_id"])
    print("MIN cos dst: ", min_dst)
    return best_match, min_dst
    
def find_people(q, plist):
    for i, p in enumerate(plist):
        if q is p['id']:
            return i
    return -1


def unique_people_search(uuf, ruf, fdb, threshold):
    """
    TODO
    input: upload unique faces, refined unique faces, face database, cosine distance threshold
    output: unique identification with correct data
    """
    known_people = []
    unidentified_people = []
    for p in uuf:
        min_dst = 1000
        best_match = None
        for f in p['person']:
            best_face_match, min_face_dst = bruteforce(f['feat'], fdb, threshold)
            if min_face_dst < min_dst:
                min_dst = min_face_dst
                best_match = best_face_match
        if best_match is not None:
            p.update({'id': best_match})
            print(min_dst)
            known_people.append(p)

    for p in ruf:
        min_dst = 1000
        best_match = None
        for f in p['person']:
            best_face_match, min_face_dst = bruteforce(f['feat'], fdb, threshold)
            if min_face_dst < min_dst:
                min_dst = min_face_dst
                best_match = best_face_match
        if best_match is not None:
            # known people
            l_id = find_people(best_match, known_people)
            if l_id < 0:
                # different people
                p.update({'id': best_match})
                known_people.append(p)
            else:
                # same person
                known_people[l_id] = merge_data(known_people[l_id], p, ['time'])
        else:
            # new people
            unidentified_people.append(p)

    return known_people, unidentified_people
