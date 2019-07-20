from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import sys

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
    crop_img = image[y:y+h, x:x+w]
#     print(image.shape)
    return image
# dimensions of our images
img_width, img_height = 320, 240

# load the model we saved
model_name = sys.argv[1]
print(model_name)
model = load_model(model_name)
model.compile(optimizer="adam", loss="mse")

# predicting images
# img = load_in_img('./dir/test - 9/black.png')#black
# img = load_in_img('./dir/test - 改-第二個轉彎後繼續前進/2019-06-13-15-15-44-5910-p059-p058.png')#black
img = load_in_img('./dir/test - 改-第二個轉彎後180度旋轉/2019-06-13-15-11-22-1631-n050-p052.png')#second turn

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print ('op classes: ')
print (classes)



