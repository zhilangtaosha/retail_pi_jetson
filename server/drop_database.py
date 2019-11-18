"""
drop face collection
"""
import shutil, os
from pymongo import MongoClient

cl = MongoClient("172.17.0.1", 27017)
db = cl["miris"]
col = db["face"]
col.drop()

shutil.rmtree("images/customer")
shutil.rmtree("data/arcface_feats")

os.mkdir("images/customer")
# os.mkdir("data/arcface_feats")