#!/usr/bin/env python
# coding: utf-8

# In[1]:


#assign the specific GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 自動增長 GPU 記憶體用量
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)
ktf.set_session(session)
graph = False

# In[2]:


# load in the data
import pandas as pd
import cv2
import numpy as np
dir_log = pd.read_csv('./deep_store.csv',usecols=[1,3])
speed_log = pd.read_csv('./deep_store.csv',usecols=[2,4])
filename_list= pd.read_csv('./deep_store.csv',usecols=[5])
#track1_log = pd.read_csv('./Dataset/driving_log.csv',names=['Center','Left','Right','Steering Angle','Throttle','Break','Speed'], skiprows=1)
#track2_log = pd.read_csv('../../data4/driving_log.csv',names=['Center','Left','Right','Steering Angle','Throttle','Break','Speed'], skiprows=1)
#track3_log = pd.read_csv('../../data5/driving_log.csv',names=['Center','Left','Right','Steering Angle','Throttle','Break','Speed'], skiprows=1)

#combine track1 and track2 data
# combine = [track_log, track1_log]
#combine = [track_log]
#make the combine ignore the index number
#track_log = pd.concat(combine, ignore_index=True) 
dir_log = np.array(dir_log).tolist()
speed_log = np.array(speed_log).tolist()
filename_list = np.array(filename_list).tolist()


## data augmentation
turn_dir_list = []
turn_speed_list = []
turn_list_img = []
straight_dir_list = []
straight_speed_list = []
straight_list_img = []
for i in range(len(dir_log)):
    #print(track_log[i][0])
    if dir_log[i][0] == 0:
        turn_dir_list.append(dir_log[i])
        turn_speed_list.append(speed_log[i])
        turn_list_img.append(filename_list[i])
    else:
        straight_dir_list.append(dir_log[i])
        straight_speed_list.append(speed_log[i])
        straight_list_img.append(filename_list[i])

turn_dir_list_ = []
turn_speed_list_= []
turn_list_img_=[]
while(len(turn_dir_list_) < len(straight_dir_list)):
    turn_dir_list_.extend(turn_dir_list)
    turn_speed_list_.extend(turn_speed_list)
    turn_list_img_.extend(turn_list_img)


dir_log = []
speed_log = []
filename_list = []

dir_log.extend(turn_dir_list_)
dir_log.extend(straight_dir_list)
speed_log.extend(turn_speed_list_)
speed_log.extend(straight_speed_list)
filename_list.extend(turn_list_img_)
filename_list.extend(straight_list_img)
#print(len(dir_log))
#print(len(speed_log))
#print(len(filename_list))

track_log = []
for i in range(len(dir_log)):
    data_frame = []
    data_frame.extend(speed_log[i])
    data_frame.extend(dir_log[i])
    data_frame.extend(filename_list[i])
    #std_data[i].extend(dir_log[i])
    #std_data[i].extend(filename_list[i])
    track_log.append(data_frame)


track_log=pd.DataFrame(track_log)
track_log.rename(columns={0:'left_wheel_speed',1:'right_wheel_speed',2:'left_wheel_dir',3:'right_wheel_dir',4:'filename'},inplace=True)
#load in the img by csv_data


def load_in_img(img_location):
    
    #folder name has save in img_location
    imageLocation = img_location
    image = cv2.imread(imageLocation,0) # Gray

    if (image is None):
        print(imageLocation)
        
    image = image[45:-9,::]
    image = cv2.resize(image, (200,200), fx=0, fy=0)
    image = image.reshape(200, 200, 1)
    # 裁切區域的 x 與 y 座標（左上角）
    x = 50
    y = 0

    # 裁切區域的長度與寬度
    w = 100
    h = 200

    # 裁切圖片
#     crop_img = image[y:y+h, x:x+w]
#     print(image.shape)
    return image

numSample = 50
centerImgs = np.array([load_in_img(img_location) for img_location in track_log['filename'][0:numSample]], dtype=np.float32)

# In[14]:


#data augmentation
import cv2
import numpy as np

# flip images horizontally
def random_flip_image(img, angle):
    
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        if angle <30:
            angle +=95
        else:
            angle -=95

    return img, angle

def modify_brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness_mul = np.random.uniform(low=0.3, high=1.0)
    img[:,:,2] = img[:,:,2]*brightness_mul
    img[:,:,2][img[:,:,2]>255] = 255
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def random_shadow(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = 200 * np.random.rand(), 0
    x2, y2 = 200 * np.random.rand(), 66
    xm, ym = np.mgrid[0:66, 0:200]
    
    # create the mask area
    mask = np.zeros_like(img[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    random_bright = np.random.uniform(low=0.2, high=0.7)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    img[:, :, 1][cond] = img[:, :, 1][cond] * random_bright
    
    img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    return img

def random_translate(img, angle, range_x=100, range_y=10):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = img.shape[:2]
    img = cv2.warpAffine(img, trans_m, (width, height))
    return img, angle


def shift_image(image):
    dx = int(160 * (np.random.rand()-0.5))
    image = np.roll(image, dx, axis=1)
    if dx>0:
        image[:, :dx] = 1
    elif dx<0:
        image[:, dx:] = 1
        
    shift_speed = int(dx / 8)
#     print(dx)
#     print(shift_speed)

    return (image, shift_speed)


def process_image(img, left_speed, left_dir, right_speed, right_dir):
#     img = modify_brightness(img)
#     img = random_shadow(img)
#     img, angle = random_flip_image(img, angle)
    #img, angle = random_translate(img, angle)
    img, shift_speed = shift_image(img)
    if left_speed > 0:
        left_speed = left_speed + shift_speed
    if left_speed < 0:
        left_speed = 0
    return img, left_speed, left_dir, right_speed, right_dir


# In[15]:


import sklearn

#load in the img and angle data
def load_data(img_sample, left_speed, left_dir, right_speed, right_dir, correction = 10):

    img_center = load_in_img(img_sample)
    img_left = load_in_img(img_sample)
    img_right =load_in_img(img_sample)
    
    left_speed_center = float(left_speed) * 1.0
    left_speed_left = left_speed_center + correction
    left_speed_right = left_speed_center - correction
    
    right_speed_center = float(right_speed) * 1
    right_speed_left = right_speed_center + correction
    right_speed_right = right_speed_center - correction

    return (img_center,img_left,img_right), (left_speed_center, left_speed_center, left_speed_center), (left_dir, left_dir, left_dir), (right_speed_center, right_speed_center, right_speed_center), (right_dir, right_dir, right_dir)

#Generator
def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            #print("in")
            batch_samples = samples[offset:offset + batch_size]   
            #print(batch_samples)
            images = []
            left_speeds = []
            left_dirs = []
            right_speeds = []
            right_dirs = []
            target = []
            for img, left_speed, left_dir, right_speed, right_dir in zip(batch_samples['filename'], batch_samples['left_wheel_speed'], batch_samples['left_wheel_dir'], batch_samples['right_wheel_speed'], batch_samples['right_wheel_dir']):
                image, left_speed, left_dir, right_speed, right_dir = load_data(img, left_speed, left_dir, right_speed, right_dir)
                for item in zip(image, left_speed, left_dir, right_speed, right_dir): #iterate camera images and steering angles
                    aug_image, augleft_speed, augleft_dir, augright_speed, augright_dir = process_image(item[0], item[1], item[2], item[3], item[4])
#                     if abs(augleft_speed[0]) or abs(augright_speed[0]) > 0.05:
                    images.append(aug_image)
                    left_speeds.append(augleft_speed)
                    left_dirs.append(augleft_dir)
                    right_speeds.append(augright_speed)
                    right_dirs.append(augright_dir)
                    target.append([augleft_speed, augleft_dir, augright_speed, augright_dir])
#                     left_speeds.append([augleft_speed])
#                     left_dirs.append([augleft_dir])
#                     right_speeds.append([augright_speed])
#                     right_dirs.append([augright_dir])
                        #left_speeds.append(aug_left_speed)
                        #print(aug_image.shape)
                
            X_train = np.array(images)
            #print(X_train)
#             y_train = np.hstack((left_speeds,left_dirs,right_speeds,right_dirs))
            y_train = np.array([left_speeds, left_dirs, right_speeds, right_dirs])
            left_speeds = np.array(left_speeds)
            left_dirs = np.array(left_dirs)
            right_speeds = np.array(right_speeds)
            right_dirs = np.array(right_dirs)
#             target = np.transpose(target)
#             print(y_train)
#             print(X_train.shape)
#             print(target.shape)
#             print(target)
            yield X_train, [left_speeds, left_dirs, right_speeds, right_dirs]
        


# In[16]:


#split the log into train_samples and validation_samples
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(track_log, test_size=0.2)
# print(train_samples)


# In[17]:


#see if generator work
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
if graph:
    iterator = generator(train_samples, 64)
    sample_images, target = iterator.__next__()

    plt.subplots(figsize=(20, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.title("speed: {:.4f}".format(target[0][i+20]))
        show_image = sample_images[i].reshape(200, 100)
        plt.imshow(show_image)
    plt.show()


# In[20]:


#NVIDIA's model
import pickle
import numpy as np
import math
from keras.utils import np_utils
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Input, Activation, LSTM, TimeDistributed
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.models import Model

INPUT_SHAPE = (200, 200, 1) # height width
DROP_PROB = 0.7
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0,input_shape=INPUT_SHAPE))
model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
model.add(Conv2D(64, 3, 3))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Dropout(DROP_PROB))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))
model.add(Dense(1))

# main_input = Lambda(lambda x: x/127.5-1.0,input_shape=INPUT_SHAPE)
main_input = Input(shape=INPUT_SHAPE, name='main_input')

x = TimeDistributed(Conv2D(filters=24,kernel_size=(3,3), activation='relu', subsample=(2, 2)))(main_input)
x = Conv2D(24, 5, 5, activation='relu', subsample=(2, 2))(main_input)
x = Conv2D(36, 5, 5, activation='relu', subsample=(2, 2))(x)
x = Conv2D(48, 5, 5, activation='relu', subsample=(2, 2))(x)
x = Conv2D(64, 3, 3, activation='relu')(x)
x = Conv2D(64, 3, 3, activation='relu')(x)
x = Dropout(DROP_PROB)(x)
x = Flatten()(x)
x = Dense(100)(x)
# x = LSTM(5, input_shape=(5,1))(x)
x = Dense(50)(x)
speed = Dense(10, name='speed')(x)
direction = Dense(10, name='direction')(x)

left_speed_output = Dense(1, name='left_speed_output')(speed)
left_dir_output_ = Dense(5, name='left_dir_output_')(speed)
left_dir_output = Dense(1, name='left_dir_output', activation='sigmoid')(left_dir_output_)
# left_dir_output = Activation.relu(alpha=0.0, threshold=0.5, name='left_dir_output')(left_dir_output_)

right_speed_output = Dense(1, name='right_speed_output')(speed)
right_dir_output_ = Dense(5,name='right_dir_output_')(speed)
right_dir_output = Dense(1,name='right_dir_output', activation='sigmoid')(right_dir_output_)
# right_dir_output = Activation.relu(alpha=0.0, threshold=0.5, name='left_dir_output')(right_dir_output_)


# In[21]:


# model summary 
model = Model(input=[main_input], output=[left_speed_output, left_dir_output, right_speed_output, right_dir_output])

model.summary()


# In[22]:


model.compile(optimizer="adam", loss="mse")


# In[27]:


import time, os, fnmatch, shutil
# export  HDF5_USE_FILE_LOCKING=FALSE
def save_model(model_name):
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
    model_name = ('./model_dir_ver4/' + model_name  + '_' + timestamp + '.h5')
    
    return model_name


# In[28]:


from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, Callback
import keras

# compile and train the model using the generator function


for index in range(40):
    train_generator = generator(train_samples, 64)
    validation_generator = generator(validation_samples, 64)
    
    history_object = model.fit_generator(
                                     train_generator, 
                                     samples_per_epoch=32000,
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=30,
                                     verbose=2)
    
    print('history_object')
    print(history_object)
    h5_output = save_model(str(history_object.history['loss'][9]) + '_' + str(index)) 
    model.save(h5_output)
    print("Model saved")
    print('Time ',index+1)

    # summarize history for loss
    plt.close('all')
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    plt.savefig("./image/gray_" + str(index) + "_image.png")
    plt.close('all')

# summarize history for loss
if graph:
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()




