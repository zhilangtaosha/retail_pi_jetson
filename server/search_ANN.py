import sys, os
sys.path.append("/sptag/SPTAG/Release")
import SPTAG
import configparser
import numpy as np
import time
from utils import cosineDistance, merge_data

class ANN(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.index_data = self.config["SPTAG"]['Index']
        self.algo = self.config["SPTAG"]['Algo']
        self.dist_method = self.config["SPTAG"]['DistMethod']
        self.data_type = self.config["SPTAG"]['DataType']
        self.dimensions = int(self.config["SPTAG"]['Dimensions'])
        self.threads = self.config["SPTAG"]['Threads']
        self.threshold = float(self.config["SPTAG"]['Threshold'])
        # if self.index_data is not None:
        if os.path.exists(self.index_data):
            self.index = SPTAG.AnnIndex.Load(self.index_data)
        else:
            # print(self.algo, self.data_type, self.dimensions)
            self.index = SPTAG.AnnIndex(self.algo, self.data_type, self.dimensions)
            self.index.SetBuildParam("NumberOfThreads", self.threads)
            self.index.SetBuildParam("DistCalcMethod", self.dist_method)
            self.build(np.ones((1, self.dimensions), dtype=np.float32), "0\n".encode())

    def build(self, data, metadata):
        if data.shape[1] != self.dimensions:
            print("Wrong data dimension", data.shape[1])
            return 1
        if self.index.BuildWithMetaData(data, metadata, data.shape[0], p_withMetaIndex=True):
        # print(data, data.shape)
        # if self.index.Build(data, data.shape[0]):
            self.index.Save(self.index_data)
            return 0
        else:
            print("failed")
        return 1

    def add(self, data, metadata):
        # print(data.shape)
        if data.shape[1] != self.dimensions:
            print("Wrong data dimension", data.shape[1])
            return 1
        if self.index.AddWithMetaData(data, metadata, data.shape[0]):
            self.index.Save(self.index_data)
            return 0
        else: 
            print("add failed")
            return 1

    def delete(self, data):
        if data.shape[1] != self.dimensions:
            print("Wrong data dimension", data.shape[1])
            return 1
        ret = self.index.Delete(data, data.shape[0])
        self.index.Save(self.index_data)
        return 0

    def search(self, data, k=3):
        ret = []
        for i in range(data.shape[0]):
            result = self.index.SearchWithMetaData(data[i], k)
            ret.append(result)
        return ret

    def find_people(self, q, plist):
        for i, p in enumerate(plist):
            if q is p['id']:
                return i
        return -1

    def face_search(self, face_clusters):
        known_people = []
        new_people = []

        for p in face_clusters:
            # best_match = None
            feats = []
            ranks = []
            for f in p['person']:
                feats.append(f['feat'])
            feats = np.asarray(feats, dtype=np.float32)
            ret = self.search(feats)

            for result in ret:
                idx, dist, metadata = result
                print(idx, dist, metadata)
                for i, d in enumerate(dist):
                    if d < self.threshold:
                        new_idx = 1
                        for r in ranks:
                            if r['idx'] == idx[i]:
                                r['dist'] = (r['dist']*r['count']+d)/(r['count']+1)
                                r['count'] += 1
                                new_idx = 0
                        if new_idx:
                            ranks.append(
                                {
                                    'idx': idx[i],
                                    'dist': d,
                                    'meta': metadata[i].decode()[:-1], # discard the '\n' at the end
                                    'count': 1
                                }
                            )
            if not len(ranks):
                # new people
                new_people.append(p)
            else:
                # known people
                ranks = sorted(
                    ranks, 
                    key=lambda i: (i['count'], i['dist']), 
                    reverse=True
                )
                print(ranks)
                l_id = self.find_people(ranks[0]['meta'], known_people)
                if l_id < 0:
                    # different people
                    p.update({'id': ranks[0]['meta']})
                    known_people.append(p)
                else:
                    # same person
                    known_people[l_id] = merge_data(known_people[l_id], p, ['time'])

        return known_people, new_people