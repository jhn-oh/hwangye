import pandas as pd
import numpy as np
import requests
from PIL import Image
import io
import cv2
import matplotlib.pyplot as plt
import random
import datetime

def get_streetview(lat, lon): #, heading_angle
    h_list = []
    total_w = 0

    for n in range(3):
        fov = "120"
        heading = 120 * n #원래는 -45
        pitch = "30"

        google_api_key = "api-key"
        url = f"https://maps.googleapis.com/maps/api/streetview?size=400x300&location={lat},{lon}&fov={fov}&heading={heading}&pitch={pitch}&key={google_api_key}"

        payload = {}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)

        # 이미지 바이트 데이터
        bytes_data = response.content

        # 이미지 변환
        img = Image.open(io.BytesIO(bytes_data))

        h_list.append(img.height)
        total_w += img.width

        if n == 0:
            img1 = img
        elif n == 1:
            img2 = img
        elif n == 2:
            img3 = img

    max_height = max(h_list)
    new_img = Image.new('RGB', (total_w, max_height))
    x_offset = 0
    for img in [img1, img2, img3]:
        new_img.paste(img, (x_offset,0))
        x_offset += img.width
    return new_img

def green_area(img):
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    hsv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 40, 40])   # HSV에서 초록색의 하한값
    upper_green = np.array([80, 255, 255]) # HSV에서 초록색의 상한값
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    green_only_image = cv2.bitwise_and(open_cv_image, open_cv_image, mask=green_mask)
    return_image = Image.fromarray(green_only_image[:, :, ::-1])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    return return_image, green_ratio

# 위도, 경도 0.0005도 단위로 하나씩 체크 (북아현동)
lat_l, lat_h = 37.556, 37.570
lon_l, lon_h = 126.947, 126.962
degree_between = 0.0005
lat_n = int((lat_h - lat_l) // (degree_between) + 1)
lon_n = int((lon_h - lon_l) // (degree_between) + 1)

df_ahyeon = np.zeros((lat_n, lon_n))

count_temp = 0

for lat_k in range(lat_n):
    lat = (10000*lat_l + 10000*lat_k*degree_between) / 10000
    for lon_k in range(lon_n):
        lon = (10000*lon_l + 10000*lon_k*degree_between) / 10000

        #사진 불러와서 분석
        streetview_image = get_streetview(lat, lon)
        return_image, green_ratio = green_area(streetview_image)
        #green_ratio = random.random()

        #return_image 로컬에 저장
        pass

        #green_ratio 모아서 데이터셋으로 만듦
        df_ahyeon[lat_k, lon_k] = green_ratio


        count_temp += 1
        print(lat, lon)

# Heatmap 생성
plt.figure(figsize=(lat_n, lon_n))
plt.imshow(df_ahyeon, cmap='Greens', interpolation='nearest', origin='lower')
plt.colorbar()
plt.title('Green View Index in Bukahyeon-dong Area')
plt.xlabel('Longtitude')
plt.ylabel("Latitude")
plt.xticks(np.arange(lon_n), labels=[f"{lon_l + j*degree_between:.4f}" for j in range(lon_n)], rotation=45)
plt.yticks(np.arange(lat_n), labels=[f"{lat_l + i*degree_between:.4f}" for i in range(lat_n)])
plt.rcParams['font.size'] = 16  # 기본 글자 크기
plt.rcParams['axes.labelsize'] = 20  # 축 레이블 글자 크기
plt.rcParams['axes.titlesize'] = 48  # 타이틀 글자 크기
plt.rcParams['xtick.labelsize'] = 12  # X축 틱 레이블 글자 크기
plt.rcParams['ytick.labelsize'] = 12  # Y축 틱 레이블 글자 크기
plt.rcParams['legend.fontsize'] = 12  # 범례 글자 크기
plt.rcParams['figure.titlesize'] = 30  # 전체 그림 타이틀 글자 크기
plt.show()