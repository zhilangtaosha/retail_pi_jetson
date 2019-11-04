import asyncio
import aiohttp
import requests
import json
import numpy as np
import cv2
import time
import base64

def b64_encode(img):
    # ret, buffer = cv2.imencode('.jpg', img, params=(cv2.IMWRITE_JPEG_QUALITY, 30))
    ret, buffer = cv2.imencode('.jpg', img)
    b64_buffer = base64.b64encode(buffer).decode('utf-8')
    return b64_buffer

def test_post(url):
    body = {
        'name': 'Hadeson',
        'description': 'test post',
        'price': 0.5,
        'tax': 0.1,
    }
    header = {"Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain"}
    body_json = json.dumps(body)
    r = requests.post(url, data=body_json, headers=header)
    print(r.json())

def large_images(url):
    s = time.time()
    img = np.zeros((112, 112, 3))
    h, w, c = img.shape
    single_img_size = h*w*c
    total_file_size = 1e6 # 1 Mb
    num_imgs = int(total_file_size / single_img_size)
    body = {
        'faces': [
            b64_encode(img)
            for i in range(num_imgs)
        ],
    }
    header = {"Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain"}
    body_json = json.dumps(body)
    print("start sending: ", time.time() - s)
    s = time.time()
    r = requests.post(url, data=body_json, headers=header)
    print(f"post {total_file_size} bytes total time:", time.time() - s)
    print(r.json())

def little_dummy(i):
    print(i)

async def dummy(wtime):
    print(f"counting dummy for {wtime}")
    await asyncio.sleep(wtime)
    for i in range(10):
        little_dummy(i)

async def loop(deadline):
    print("start loop")
    lst = time.time()
    while(True):
        if time.time() - lst > deadline:
            break
        # await asyncio.sleep(0.5)

async def large_images_sync(url):
    s = time.time()
    img = np.zeros((112, 112, 3))
    h, w, c = img.shape
    single_img_size = h*w*c
    total_file_size = 4e6 # 45 Mb
    num_imgs = int(total_file_size / single_img_size)

    body = {
        'unique_faces': [
            {
                'faces': [
                    b64_encode(img)
                    for i in range(np.random.randint(3) + 1)
                ],
                'time': [
                    time.time()
                    for i in range(np.random.randint(100) + 1)
                ]
            }
            for j in range(3)
        ],
        'raw_faces': [
            {
                'face': b64_encode(img),
                'time': time.time()
            }
            for i in range(num_imgs)
        ],
    }
    header = {"Content-type": "application/x-www-form-urlencoded",
            "Accept": "text/plain"}
    body_json = json.dumps(body)
    print("prep data: ", time.time() - s)
    print("start sending")
    s = time.time()
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, data=body_json, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
                data = await response.json()
                print(data)
        except asyncio.TimeoutError:
            print("Connection Timeout")
        print(f"post {total_file_size} bytes total time:", time.time() - s)
    # async with requests.post(url, data=body_json, headers=header) as response:
    #     data = await response.json()
    #     print(f"post {total_file_size} bytes total time:", time.time() - s)
    #     print(data)

# test_post("http://127.0.0.1:80/items/")
# test_post("http://192.168.11.240:80/items/")
# large_images("http://192.168.11.240:80/face/log/")
# asyncio.run(large_images_sync("http://192.168.11.240:80/face/log/"))
async def main():
    st = time.time()
    print("start")
    await asyncio.gather(
        large_images_sync("http://192.168.11.240:80/face/log/"),
        dummy(0.01),
        loop(0.01)
        # (print(i) for i in range(10)),
    )

    # await asyncio.wait([
    #     dummy(),
    #     large_images_sync("http://192.168.11.240:80/face/log/"),
    #     ],
    #     return_when=asyncio.FIRST_COMPLETED
    # )

    # task1 = asyncio.create_task(large_images_sync("http://192.168.11.240:80/face/log/"))
    # task2 = asyncio.create_task(dummy(2))
    # task3 = asyncio.create_task(loop(3))
    # await task1
    # await task2
    # await task3

    print("end", time.time() - st)

# for gather
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())

# for create_task
asyncio.run(main())
asyncio.run(loop(1))
