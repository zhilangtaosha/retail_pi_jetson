"""
communication with server
"""
import configparser
import requests, json

class Xnet(object):
    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = configparser.ConfigParser()
            self.config.read("config.ini")
        self.server_ip = self.config["XNET"]['Server_ip']
        self.face_logging_service = self.server_ip + self.config["XNET"]['Face_logging']
        self.timeout = float(self.config["XNET"]['Timeout'])

    def log_face(self, unique_faces):
        """
        log several faces of each unique person and time face appear in frame
        """
        header = {"Content-type": "application/x-www-form-urlencoded",
                "Accept": "text/plain"} 
        body_json = json.dumps(unique_faces)
        # hacky way to post without receiving response
        print(self.face_logging_service)
        # requests.post(
        #     self.face_logging_service, 
        #     data=body_json, 
        #     headers=header
        # )
        try:
            requests.post(
                self.face_logging_service, 
                data=body_json, 
                headers=header,
                timeout=self.timeout
            )
        except:
            pass
