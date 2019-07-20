#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import json
import sys
import pickle
from moviepy.editor import *
from IPython.display import HTML 
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from keras.preprocessing import image


# In[2]:


# initialize
path="./test_dir/"
# path_list=os.listdir(path)
#path_list.sort()
json_dic = []
clips = []
# graphic
time_index = 1
time_series = []


# In[3]:


# for dirname in path_list:
#     file_path = os.path.join(path, dirname)
#     if(os.path.isdir(file_path)):
#         print(dirname)
#         dirname=os.listdir(path + '/' + dirname)
#         dirname.sort()
#         for filename in dirname:
#             image = cv2.imread(os.path.join(file_path,filename), 0)
#             clips.append(image)


# In[18]:


# dimensions of our images
img_width, img_height = 200, 200

def mod_img(image):
    image = image[45:-9,::]
    image = cv2.resize(image, (img_height,img_width), fx=0, fy=0)
    image = image.reshape(img_height, img_width, 1)
    # 裁切區域的 x 與 y 座標（左上角）
    x = 50
    y = 0

    # 裁切區域的長度與寬度
    w = 100
    h = 200

    # 裁切圖片
    crop_img = image[y:y+h, x:x+w]
    return image


# In[22]:


from keras.preprocessing import image
def handle_dir(left, right):
#     return left, right
    if left < 0.65:
        return 0, 1, left, right
    return 1, 1, left, right

def handle_img(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_mod = mod_img(gray_image)
    x = image.img_to_array(image_mod)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    left_dir, right_dir, left, right = handle_dir(classes[1][0][0], classes[3][0][0])
    cv2.putText(img, "left_wheel_speed: " + str(int(classes[0][0][0]) + 10), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "left_wheel_dir: " + str(left_dir) + '(' + str(left) + ')', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "right_wheel_speed: " + str(int(classes[2][0][0]) + 10), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "right_wheel_dir: " + str(right_dir) + '(' + str(right) + ')', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    return img

    


# In[6]:


model_name = './model_dir_0_1/5.885818289764096_15_Jun-06-2019_03-39-41.h5'
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
print(model_name)
model = load_model(model_name)
model.compile(optimizer="adam", loss="mse")


# In[7]:


# video_clip_output = 'clip_predict.mp4'
# video = ImageSequenceClip(list(clips), fps=15).resize(1)
# video.write_videofile(video_clip_output, fps=15)


# In[23]:


test_clip_filename = './clip.mp4'
new_clip_output = './predict.mp4'
test_clip = VideoFileClip(test_clip_filename)
new_clip = test_clip.fl_image(lambda x: handle_img(x)) #NOTE: this function expects color images!!
new_clip.write_videofile(new_clip_output,audio=False, fps = 2)


# # In[24]:


# HTML("""
#             <video width="512" height="424" controls>
#               <source src="{0}" type="video/mp4">
#             </video>
#             """.format(new_clip_output))


# # In[10]:


# left_wheel_dir = []
# left_wheel_speed = []
# right_wheel_dir = []
# right_wheel_speed = []
# filename = []

# for data in json_dic:
#     left_wheel_dir.append(data['left_wheel_dir'])
#     left_wheel_speed.append(data['left_wheel_speed'])
#     right_wheel_dir.append(data['right_wheel_dir'])
#     right_wheel_speed.append(data['right_wheel_speed'])
#     filename.append(data['filename'])
    
# plt.subplot(2,1,1) 
# plt.plot(time_series, left_wheel_speed)

# plt.subplot(2,1,2)  
# plt.plot(time_series, right_wheel_speed)

# plt.show()


# # In[11]:


# # array to pd data type
# df = pd.DataFrame.from_dict([])
# csv_file_name = 'store.csv'

# df['left_wheel_dir'] = left_wheel_dir
# df['left_wheel_speed'] = left_wheel_speed
# df['right_wheel_dir'] = right_wheel_dir
# df['right_wheel_speed'] = right_wheel_speed
# df['filename'] = filename

# # print (df)
# df.to_csv(csv_file_name, sep=',', encoding='utf-8')


# In[ ]:




