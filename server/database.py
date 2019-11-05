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
        face_col_name = self.config["MONGO"]['FaceCollection']
        log_col_name = self.config["MONGO"]['LogCollection']
        self.customer_img_dir = self.config["IMG_DIR"]['Customer']
        self.employee_img_dir = self.config["IMG_DIR"]['Employee']
        self.log_img_dir = self.config["IMG_DIR"]['Log']
        self.client = MongoClient(url, port)
        self.db = self.client[db_name]
        self.face_collection = self.db[face_col_name]
        self.log_collection = self.db[log_col_name]

    def loadFaces(self):
        """
        return cursor and collection of face database
        """
        # get the whole collection
        people = list(self.face_collection.find())
        return people

    def loadLogs(self):
        """
        return cursor and collection of log database
        """
        logs = list(self.log_collection.find())
        return logs

    def newPeopleLog(self, known_people, new_people):
        """
        log new people
        """
        for p in known_people:
            new_log = {
                'face_id': p['id'],
                'time': p['time']
            }
            p_id = self.log_collection.insert_one(new_log).inserted_id
            self.log_collection.update_one({'_id': p_id}, {"$set": new_log}, upsert=False)
            img_name = f"{int(time.time())}_{str(p['id'])}.jpg"
            img_path = os.path.join(self.log_img_dir, img_name)
            # choose first face for each person. TODO: complicated saving stuff go here
            cv2.imwrite(img_path, p['person'][0]['face'])

        for p in new_people:
            new_log = {
                'face_id': p['id'],
                'time': p['time']
            }
            p_id = self.log_collection.insert_one(new_log).inserted_id
            self.log_collection.update_one({'_id': p_id}, {"$set": new_log}, upsert=False)
            img_name = f"{int(time.time())}_{str(p['id'])}.jpg"
            img_path = os.path.join(self.log_img_dir, img_name)
            # choose first face for each person. TODO: complicated saving stuff go here
            cv2.imwrite(img_path, p['person'][0]['face'])
        return 0

    def addNewFaces(self, new_people):
        """    
        update database with new people data
        """    
        for j, np in enumerate(new_people):
            new_face = {
                'feats': [np['person'][i]['feat'].tolist() for i in range(len(np['person']))],
                'age': np['age'],
                'gender': np['gender'],
                'create_time': time.time(),
            }
            # insert to Mongo
            p_id = self.face_collection.insert_one(new_face).inserted_id
            # commit changes
            self.face_collection.update_one({'_id': p_id}, {"$set": new_face}, upsert=False)
            # create account imgs dir
            new_id = str(p_id)
            print("new customer: ", new_id)
            img_dir = os.path.join(self.customer_img_dir, new_id)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            for i, p in enumerate(np['person']):
                img_path = os.path.join(img_dir, f"{i}.jpg")
                cv2.imwrite(img_path, p['face'])
            new_people[j].update({'id': new_id})
        return new_people
