from pymongo import MongoClient
import configparser

class FaceDatabase(object):
    def __init__(self, parent=None):
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        # url = self.config["MONGO"]['Url']
        # port = int(self.config["MONGO"]['Port'])
        # db_name = self.config["MONGO"]['Database']
        # col_name = self.config["MONGO"]['FaceCollection']
        # self.client = MongoClient(url, port)
        # self.db = self.client[db_name]
        # self.collection = self.db[col_name]

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

    def newFaces(self, unique_people):
        """    
        TODO
        update database with new people data
        """    
        return 0
