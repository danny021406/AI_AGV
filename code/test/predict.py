from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
import sys
from canny_line import color_frame_process

def load_in_img(img_location):
    
    #folder name has save in img_location
    imageLocation = img_location
    image = cv2.imread(imageLocation) # Gray

    if (image is None):
        print(imageLocation)
#     print(imageLocation)

#     image = image[45:-9,::]
    image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_CUBIC)
    image = color_frame_process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.reshape(200, 200, 1)
    return image

# dimensions of our images
img_width, img_height = 320, 240
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

# load the model we saved
model_name = sys.argv[1]
print(model_name)
model = load_model(model_name)
model.compile(optimizer="adam", loss="mse")

# predicting images
# img = load_in_img('./dir/test - 9/black.png')#black
# img = load_in_img('./dir/test - 改-第二個轉彎後繼續前進/2019-06-13-15-15-44-5910-p059-p058.png')#black
img = load_in_img('../../data/103/test - 1/2019-10-03-14-38-47-9333-p077-p080.png')#second turn

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print ('op classes: ')
print (classes)



