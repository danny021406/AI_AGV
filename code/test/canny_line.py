import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 255, 255], thickness=3):
    left_line = 0            
    right_line = 200  
    x11 = 0
    x12 = 0
    x21 = 0
    x22 = 0
    if(not lines is None):
        for line in lines:
            for x1,y1,x2,y2 in line:
    #             cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                if((x2 - x1) != 0 ):
                    if 0.1 <= np.abs((y2 - y1)/(x2 - x1)) <= 2:
                        if (x1 + x2) / 2 < 100:
                            if left_line < (x1 + x2) / 2:
                                left_slope = (y2 - y1)/(x2 - x1)
                                x11 = int(x1 - (y1 - 50) / left_slope)
                                x12 = int(x1 - (y1 - 200) / left_slope)
                                left_line = (x1 + x2) / 2
                        else:
                            if right_line > (x1 + x2) / 2:
                                right_slope = (y2 - y1)/(x2 - x1)
                                x21 = int(x1 - (y1 - 50) / right_slope)
                                x22 = int(x1 - (y1 - 200) / right_slope)
                                right_line = (x1 + x2) / 2
        if x11 != 0 | x12 != 0:
            cv2.line(img, (int(x11), 50), (int(x12), 200),color, thickness)
        if x21 != 0 | x22 != 0: 
            cv2.line(img, (int(x21), 50), (int(x22), 200),color, thickness)
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., z=0.):
    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))

    return cv2.addWeighted(initial_img, a, img, b, z)

def color_frame_process(image):


    # Read in and grayscale the image
    gray = grayscale(image)
#     plt.imshow(gray)
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray,kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 80
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    
    # This time we are defining a four sided polygon to mask
#     imshape = image.shape
#     vertices = np.array([[(0,imshape[0]),(10, 200), (190, 200), (imshape[1],imshape[0])]], dtype=np.int32)
#     masked_edges = region_of_interest(edges, vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180# angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 50    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    line_image_ROI_test = np.copy(image)*0 # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(edges, rho, theta, threshold,min_line_length, max_line_gap)

    img_blend = weighted_img(line_image,image , a=0.8, b=1., z=0.)

    return img_blend