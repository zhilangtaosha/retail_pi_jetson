from pymongo import MongoClient
import time
import os
import cv2
import configparser

class FaceDatabase(object):
    def __init__(self, parent=None):
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        url = self.config["MONGO"]['Url']
        port = int(self.config["MONGO"]['Port'])
        db_name = self.config["MONGO"]['Database']
        col_name = self.config["MONGO"]['FaceCollection']
        self.customer_img_dir = self.config["IMG_DIR"]['Customer']
        self.employee_img_dir = self.config["IMG_DIR"]['Employee']
        self.log_img_dir = self.config["IMG_DIR"]['Log']
        self.client = MongoClient(url, port)
        self.db = self.client[db_name]
        self.collection = self.db[col_name]

    def loadFaces(self):
        """
        return cursor and collection of face database
        """
        # get the whole collection
        people = list(self.collection.find())
        return people

    def loadLogs(self):
        """
        TODO
        return cursor and collection of log database
        """
        return 0

    def newPeopleLog(self, unique_people):
        """
        TODO
        log new people
        """
        return 0

    def addNewFaces(self, new_people):
        """    
        TODO
        update database with new people data
        """    
        for np in new_people:
            new_face = {
                'feats': [np['person'][i]['feat'].tolist() for i in range(len(np['person']))],
                'age': np['age'],
                'gender': np['gender'],
                'create_time': time.time(),
            }
            # insert to Mongo
            p_id = self.collection.insert_one(new_face).inserted_id
            # commit changes
            self.collection.update_one({'_id': p_id}, {"$set": new_face}, upsert=False)
            # create account imgs dir
            new_id = str(p_id)
            print("new customer: ", new_id)
            img_dir = os.path.join(self.customer_img_dir, new_id)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            for i, p in enumerate(np['person']):
                img_path = os.path.join(img_dir, f"{i}.jpg")
                cv2.imwrite(img_path, p['face'])
        return 0
