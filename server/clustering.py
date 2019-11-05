import numpy as np
from sklearn.cluster import DBSCAN

def face_clustering(np_feats):
    """
    get clustered faces based on embedded feature
    """
    feat_num, feat_size = np_feats.shape
    if feat_num == 1:
        return np.asarray([0])
    print(feat_num, feat_size)
    np_feats = np.squeeze(np_feats)
    print(np_feats.shape)
    cluster = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(np_feats)
    return cluster.labels_

def cluster_raw_faces(feats, raw_data, max_img_per_person=3):
    """
    return list of unique faces based on raw data
    """
    unique_faces = []
    print(feats)
    feat_num, feat_size = feats.shape
    # clustering
    if feat_num == 0:
        return None
    labels = face_clustering(feats)
    class_ids = np.unique(labels)
    print("unique ", class_ids)
    print(labels)
    for cli in class_ids:
        if cli == -1:
            # noise
            continue
        if len(labels) > 1:
            cli_feat_ids = np.asarray(np.where(labels==cli))
            cli_feat_ids = np.squeeze(cli_feat_ids)
            sample_size = cli_feat_ids.shape[0]
            num_upload_imgs = min(max_img_per_person, sample_size)
            chosen_ids = np.unique(
                np.random.choice(
                    sample_size,
                    num_upload_imgs,
                    replace=False
                )
            )
        else:
            cli_feat_ids = np.asarray([0])
            chosen_ids = np.asarray([0])
        unique_faces.append(
            {
                'person': [
                    {
                        'face': raw_data[cli_feat_ids[i]]['face'],
                        'feat': feats[cli_feat_ids[i]],
                    }
                    for i in chosen_ids
                ], 
                'time': [
                    raw_data[i]['time']
                    for i in cli_feat_ids
                ],
            }
        )
    print("num of unique people: ", len(unique_faces))

    return unique_faces