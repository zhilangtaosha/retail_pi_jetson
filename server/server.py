"""
simple POC server for:
 - receive face images, timestamp from Pis
 - classify faces into old customer/new customer/employee/...
 - facial analysis: age, gender analysis for new customer
 - receive timely heatmap update from Nanos
"""
import os
import time
from datetime import datetime
import json, base64, io
import logging 
import configparser
import cv2
import numpy as np
from bson import ObjectId
from pymongo import MongoClient
from sys import argv
from http.server import BaseHTTPRequestHandler, HTTPServer

class S(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.customer_img_dir = self.config["IMG_DIR"]["Customer"]
        self.employee_img_dir = self.config["IMG_DIR"]["Employee"]
        self.face_logging = self.config["SERVICE"]['Face_logging']
        super().__init__(*args, **kwargs)

    def do_OPTIONS(self):           
        self.send_response(200, "ok")       
        self.send_header('Access-Control-Allow-Origin', '*')                
        self.send_header("Access-Control-Allow-Credentials", "true")                
        self.send_header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Range, Content-Disposition, Authorizaion, Access-Control-Allow-Headers, Origin, Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers") 
        self.end_headers()
        response = {
            'error_code': 0
        }
        response_js = json.dumps(response)
        self.wfile.write(response_js.encode('utf-8'))

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def loadLogData(self):
        """
        return collection
        """
        url = self.config["MONGO"]['Url']
        port = int(self.config["MONGO"]['Port'])
        db_name = self.config["MONGO"]['Database']
        col_name = self.config["MONGO"]['LogCollection']
        client = MongoClient(url, port)
        db = client[db_name]
        collection = db[col_name]
        # get the whole collection
        # logs = list(collection.find())
        return collection

    def face_analysis(self, data_dict):
        """
        analyze uploaded faces
        """
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for i, person in enumerate(data_dict):
            faces = person['faces']
            for face in faces:
                face_bin = base64.b64decode(face)
                face_stream = io.BytesIO(face_bin)
                face_cv = cv2.imdecode(np.fromstring(face_stream.read(), np.uint8), 1)
                img_name = "{}_{}{}".format(now, i, ".jpg")
                img_name = img_name.replace("-", "_")
                img_name = img_name.replace(":", "_")
                img_name = img_name.replace(" ", "_")
                img_path = os.path.join(self.customer_img_dir, img_name)
                cv2.imwrite(img_path, face_cv)
        self._set_response()
        response = {
            'error_code': 0
        }
        response_js = json.dumps(response)
        self.wfile.write(response_js.encode('utf-8'))
        return 0

    # def updateLog(self, new_log, face_img):
    #     p_id = self.log_collection.insert_one(new_log).inserted_id
    #     self.log_collection.update_one({'_id': p_id}, {"$set": new_log}, upsert=False)
    #     new_img_name = "{}_{}.jpg".format(new_log['time'], new_log['result'])
    #     # remove special chars
    #     new_img_name = new_img_name.replace("-", "_")
    #     new_img_name = new_img_name.replace(":", "_")
    #     new_img_name = new_img_name.replace(" ", "_")
    #     new_img_path = os.path.join(self.log_img_dir, new_img_name)
    #     # print(new_img_path)
    #     cv2.imwrite(new_img_path, face_img)
    #     return p_id

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n",
                     str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(
            self.path).encode('utf-8'))


    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        # print(post_data)
        data_dict = json.loads(post_data.decode('utf-8'))
        # data_dict = json.loads(post_data) # use this if no utf encode is used
        print(content_length)
        if self.path == self.face_logging:
            self.face_analysis(data_dict)


def run(server_class=HTTPServer, handler_class=S, port=9999):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    print(server_address)
    # global face_recog
    # face_recog = FaceRecognition()
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()