"""
communication with server
"""
import configparser
import requests, json
import time
import aiohttp
import asyncio

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

    # def log_face_request(self, unique_faces):
    #     """
    #     log several faces of each unique person and time face appear in frame
    #     """
    #     header = {"Content-type": "application/x-www-form-urlencoded",
    #             "Accept": "text/plain"} 
    #     body_json = json.dumps(unique_faces)
    #     # hacky way to post without receiving response
    #     print(self.face_logging_service)
    #     # requests.post(
    #     #     self.face_logging_service, 
    #     #     data=body_json, 
    #     #     headers=header
    #     # )
    #     try:
    #         requests.post(
    #             self.face_logging_service, 
    #             data=body_json, 
    #             headers=header,
    #             timeout=self.timeout
    #         )
    #     except:
    #         pass

    async def log_face(self, unique_faces, unprocess_faces):
        """
        log several faces of each unique person and time face appear in frame
        using aiohttp
        """
        body_json = json.dumps({
            'unique_faces': unique_faces,
            'raw_faces': unprocess_faces,
            'time': time.time()
        })
        print(self.face_logging_service)
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post( self.face_logging_service, data=body_json) as response:
                    data = await response.json() 
                    print(data)
            except asyncio.TimeoutError:
                # TODO: save data to disk
                print("Connection Timeout")
        return data