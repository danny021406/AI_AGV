# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import cv2
import glob
import numpy as np
import time

num_of_extend = 1
def shift_image(image):
    dx = int(160 * (np.random.rand()-0.5))
    image = np.roll(image, dx, axis=1)
    if dx>0:
        image[:, :dx] = 1
    elif dx<0:
        image[:, dx:] = 1
        
    shift_speed = int((dx * 3) / 16)
    print(dx)
    print(shift_speed)

    return (image, shift_speed)

def couting_area(image):
    image_area = 0
    start = time.clock()
    kernel_size = (9, 9)
    sigma = 1.5
    

    image = cv2.GaussianBlur(image, kernel_size, sigma)
#     ret,thresh = cv2.threshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),127,255,0)
    contours,hierarchy = cv2.findContours(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#得到轮廓信息
    image = cv2.drawContours(image,contours,-1,(0,0,255),3) 
#     print(len(contours))
    for i in range(len(contours)):
        cnt = contours[i]#取第一条轮廓
        area = cv2.contourArea(cnt)
        if(area > 0):
            image_area = image_area + area
    end =  time.clock()
    height, width, channels = image.shape
    print("Image area: ", image_area)
#     print(height)
#     print(width)
#     print(height * width)
    print(image_area / (height * width))
    print("CPU Time: ", end - start)
    return image

def rotating_image(image):
    (h,w) = image.shape[:2]
    center = (w / 2,h / 2)
    M = cv2.getRotationMatrix2D(center,45,1)
    rotated = cv2.warpAffine(image,M,(w,h))
    return rotated

def main():
#     ./dir/test - 1/2019-05-24-11-44-35-5233-p106-p110.png
# 2019-05-24-11-45-33-2530-n028-p028.png
    image = cv2.imread('./dir/test - 改-第二個轉彎後繼續前進/2019-06-13-15-15-44-5910-p059-p058.png')
#     image_shift, speed = shift_image(image)
    image_shift = couting_area(image)
    cv2.imwrite('./test.png', image_shift)
#     datagen = ImageDataGenerator(rotation_range=180,
#                                  width_shift_range=0,		
#                                  height_shift_range=0,	
#                                  #shear_range=0.1,		
#                                  #zoom_range=0.0,			
#                                  #channel_shift_range=20,		
#                                  horizontal_flip=True,		
#                                  vertical_flip = True,		
#                                  fill_mode='nearest')			
#     count = 1
#     for img in glob.glob('./label/*'):
#         #print(img)
#         count += 1
#         save_path = './output/'#img[:len(img)-4]+'_'+ str(count)
#         #print(save_path)
#         im = Image.open(img)
#         im = im.convert('RGB')
#         im = np.asarray(im, dtype='uint8')
#         arr = im.reshape(1,im.shape[0],im.shape[1],3)							
#         i = 0
#         for batch in datagen.flow(arr, batch_size=1,
#                           save_to_dir = save_path , save_prefix = img[8:len(img)-4], save_format='jpg'):
#             i += 1
#             if i >= num_of_extend:										
#                 i =0
#                 break

if __name__ == '__main__':
    main()
    

'''
rotation_range: Int. Degree range for random rotations. 
width_shift_range: Float, 1-D array-like or int
    - float: fraction of total width, if < 1, or pixels if >= 1.
    - 1-D array-like: random elements from the array.
    - int: integer number of pixels from interval
        `(-width_shift_range, +width_shift_range)`
    - With `width_shift_range=2` possible values
        are integers `[-1, 0, +1]`,
        same as with `width_shift_range=[-1, 0, +1]`,
        while with `width_shift_range=1.0` possible values are floats in
        the interval [-1.0, +1.0).
height_shift_range: Float, 1-D array-like or int
    - float: fraction of total height, if < 1, or pixels if >= 1.
    - 1-D array-like: random elements from the array.
    - int: integer number of pixels from interval
        `(-height_shift_range, +height_shift_range)`
    - With `height_shift_range=2` possible values
        are integers `[-1, 0, +1]`,
        same as with `height_shift_range=[-1, 0, +1]`,
        while with `height_shift_range=1.0` possible values are floats in
        the interval [-1.0, +1.0).
shear_range: Float. Shear Intensity
    (Shear angle in counter-clockwise direction in degrees)
zoom_range: Float or [lower, upper]. Range for random zoom.
    If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
channel_shift_range: Float. Range for random channel shifts.
fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
    Default is 'nearest'.
    Points outside the boundaries of the input are filled
    according to the given mode:
    - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
    - 'nearest':  aaaaaaaa|abcd|dddddddd
    - 'reflect':  abcddcba|abcd|dcbaabcd
    - 'wrap':  abcdabcd|abcd|abcdabcd
cval: Float or Int.
    Value used for points outside the boundaries
    when `fill_mode = "constant"`.
horizontal_flip: Boolean. Randomly flip inputs horizontally.
vertical_flip: Boolean. Randomly flip inputs vertically.
'''
