"""
drop face collection
"""
from pymongo import MongoClient

cl = MongoClient("127.0.0.1", 27017)
db = cl["miris"]
col = db["face"]
col.drop()