class LaneDetection:
    def __init__(self,
                 buffer_size=10,
                 kernel_size=3,
                 canny_lo_thresh=20,
                 canny_hi_thresh=60,
                 hough_rho=2,
                 hough_theta=(np.pi/180),
                 hough_thresh=50,
                 hough_min_line_len=15,
                 hough_max_line_gap=10,
                 min_line_slope=0.5,
                 max_line_slope=0.9,
                 line_color=[255, 0, 0]):
        self.buffer_size = buffer_size
        self.kernel_size = kernel_size
        self.canny_lo_thresh = canny_lo_thresh
        self.canny_hi_thresh = canny_hi_thresh
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_thresh = hough_thresh
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        self.min_line_slope = min_line_slope
        self.max_line_slope = max_line_slope
        self.line_color = line_color
        self.left_buffer = []
        self.right_buffer = []

    def filter_colors_hsv(self, img):
        """
        Convert image to HSV color space and suppress any colors
        outside of the defined color ranges
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        yellow_dark = np.array([15, 127, 127], dtype=np.uint8)
        yellow_light = np.array([25, 255, 255], dtype=np.uint8)
        yellow_range = cv2.inRange(img, yellow_dark, yellow_light)

        white_dark = np.array([0, 0, 200], dtype=np.uint8)
        white_light = np.array([255, 30, 255], dtype=np.uint8)
        white_range = cv2.inRange(img, white_dark, white_light)

        return cv2.bitwise_and(img, img, mask=(yellow_range | white_range))

    def get_masked_image(self, img):
        """
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        rows, cols = img.shape[:2]
        top = int(rows/2 + 50)
        vertices = np.array([[(      30, rows),
                              (     450,  top),
                              (cols-450,  top),
                              ( cols-30, rows)]], dtype=int)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        return img & mask

    def add_line_to_buffer(self, line, buffer):
        """
        Push the current line onto this buffer and drop the oldest line
        if there are more lines than the specified buffer size
        """
        buffer.append(line)
        return buffer[-self.buffer_size:]

    def get_line_segment(self, x1, x2, line):
        """
        Use the slope and y-intercept values of the line to derive
        the y1,y2 values for the inputs x1,x2
        """
        fx = np.poly1d(line)
        y1 = int(fx(x1))
        y2 = int(fx(x2))
        return ((x1,y1), (x2,y2))
    
    def get_line_intersection(self, left_line, right_line):
        """
        find the intersection of 2 lines
        """
        left_slope, left_intercept = left_line
        right_slope, right_intercept = right_line
        
        # put the coordinates into homogeneous form
        a = [[left_slope, -1],
             [right_slope, -1]]
        b = [-left_intercept, -right_intercept]
        x, y = np.linalg.solve(a, b)
        return int(x)
    
    def partition_line_segments(self, line_segments, mid_x):
        """
        Separates line segments by their position in the image to determine which is the
        left line vs. the right line. Filter out line segments with slopes outside a
        given minimum / maxiumum
        """
        left_points = {'X': [], 'Y': [],}
        right_points = {'X': [], 'Y': [],}
        for segment in line_segments:
            x1, y1, x2, y2 = segment[0]
            dY = y2 - y1
            dX = x2 - x1
            if dX != 0: # don't divide by zero
                slope = float(dY) / float(dX)
                if x1 < mid_x and x2 < mid_x: # left lines
                    if -self.max_line_slope < slope < -self.min_line_slope:
                        left_points['X'] += [x1,x2]
                        left_points['Y'] += [y1,y2]
                elif x1 > mid_x and x2 > mid_x: # right lines
                    if self.max_line_slope > slope > self.min_line_slope:
                        right_points['X'] += [x1,x2]
                        right_points['Y'] += [y1,y2]
        return left_points, right_points
    
    def fit_lines_to_points(self, left_points, right_points):
        """
        fit a line (slope, y-intercept) to each of the point sets and add to buffer
        return the average slope and y-intercept values over the last N frames
        """
        if len(left_points['X']) > 1:
            left_line = np.polyfit(left_points['X'], left_points['Y'], 1)
            self.left_buffer = self.add_line_to_buffer(left_line, self.left_buffer)
        if len(right_points['X']) > 1:
            right_line = np.polyfit(right_points['X'], right_points['Y'], 1)
            self.right_buffer = self.add_line_to_buffer(right_line, self.right_buffer)
        
        return (np.mean(self.left_buffer, axis=0),
                np.mean(self.right_buffer, axis=0),)

    def get_lane_lines(self, img):
        left_x = 0
        right_x = img.shape[1]
        line_segments = cv2.HoughLinesP(img, self.hough_rho, self.hough_theta, self.hough_thresh,
                                        minLineLength=self.hough_min_line_len,
                                        maxLineGap=self.hough_max_line_gap)
        left_points, right_points = self.partition_line_segments(line_segments, int(right_x / 2))
        left_line, right_line = self.fit_lines_to_points(left_points, right_points)
        intersection_x = self.get_line_intersection(left_line, right_line)

        return (self.get_line_segment(left_x, intersection_x, left_line),
                self.get_line_segment(right_x, intersection_x, right_line),)
    
    def get_filtered_masked_image(self, img):
        filtered_img = self.filter_colors_hsv(img)
        filtered_img = cv2.GaussianBlur(filtered_img[:,:,2], (self.kernel_size, self.kernel_size), 0)
        filtered_img = cv2.Canny(filtered_img, self.canny_lo_thresh, self.canny_hi_thresh)
        return self.get_masked_image(filtered_img)
    
    def process_image(self, img):
        filtered_img = self.get_filtered_masked_image(img)
        lane_lines = self.get_lane_lines(filtered_img)
        line_img = np.zeros_like(img)
        for line in lane_lines:
            cv2.line(line_img, line[0], line[1], self.line_color, 5)
        return cv2.addWeighted(img, 1., line_img, 1., 0.)