import cv2
import numpy as np
from time import time
import random
import math

#initializing font for puttext
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#loading apple image and making its mask to overlay on the video feed
apple = cv2.imread("Apple-Fruit-Download-PNG.png",-1)
apple_mask = apple[:,:,3]
apple_mask_inv = cv2.bitwise_not(apple_mask)
apple = apple[:,:,0:3]
# resizing apple images
apple = cv2.resize(apple,(40,40),interpolation=cv2.INTER_AREA)
apple_mask = cv2.resize(apple_mask,(40,40),interpolation=cv2.INTER_AREA)
apple_mask_inv = cv2.resize(apple_mask_inv,(40,40),interpolation=cv2.INTER_AREA)
#initilizing a black blank image
blank_img = np.zeros((480,640,3),np.uint8)
#capturing video from webcam
video = cv2.VideoCapture(0)
#kernels for morphological operations
kernel_erode = np.ones((4,4),np.uint8)
kernel_close = np.ones((15,15),np.uint8)
#for blue [99,115,150] [110,255,255]
#function for detecting red color
def detect_red(hsv):
    #lower bound for red color hue saturation value
    lower = np.array([136, 87, 111])  # 136,87,111
    upper = np.array([179, 255, 255])  # 180,255,255
    mask1 = cv2.inRange(hsv, lower, upper)
    lower = np.array([0, 110, 100])
    upper = np.array([3, 255, 255])
    mask2 = cv2.inRange(hsv, lower, upper)
    maskred = mask1 + mask2
    maskred = cv2.erode(maskred, kernel_erode, iterations=1)
    maskred = cv2.morphologyEx(maskred,cv2.MORPH_CLOSE,kernel_close)
    return maskred

#functions for detecting intersection of line segments.
def orientation(p,q,r):
    val = int(((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))
    if val == 0:
        #linear
        return 0
    elif (val>0):
        #clockwise
        return 1
    else:
        #anti-clockwise
        return 2

def intersect(p,q,r,s):
    o1 = orientation(p, q, r)
    o2 = orientation(p, q, s)
    o3 = orientation(r, s, p)
    o4 = orientation(r, s, q)
    if(o1 != o2 and o3 != o4):
        return True

    return False

#initilizing time (used for increasing the length of snake per second)
start_time = int(time())
# q used for intialization of points
q,snake_len,score,temp=0,200,0,1
# stores the center point of the red blob
point_x,point_y = 0,0
# stores the points which satisfy the condition, dist stores dist between 2 consecutive pts, length is len of snake
last_point_x,last_point_y,dist,length = 0,0,0,0
# stores all the points of the snake body
points = []
# stores the length between all the points
list_len = []
# generating random number for placement of apple image
random_x = random.randint(10,550)
random_y = random.randint(10,400)
#used for checking intersections
a,b,c,d = [],[],[],[]
#main loop
while 1:
    xr, yr, wr, hr = 0, 0, 0, 0
    _,frame = video.read()
    frame = cv2.flip(frame,1)
    # initilizing the accepted points so that they are not at the top left corner
    if(q==0 and point_x!=0 and point_y!=0):
        last_point_x = point_x
        last_point_y = point_y
        q=1
    #converting to hsv
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maskred = detect_red(hsv)
    #finding contours
    _, contour_red, _ = cv2.findContours(maskred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #drawing rectangle around the accepted blob
    try:
        for i in range (0,10):
            xr, yr, wr, hr = cv2.boundingRect(contour_red[i])
            if (wr*hr)>2000:
                break
    except:
        pass
    cv2.rectangle(frame, (xr, yr), (xr + wr, yr + hr), (0, 0, 255), 2)
    #making snake body
    point_x = int(xr+(wr/2))
    point_y = int(yr+(hr/2))
    # finding distance between the last point and the current point
    dist = int(math.sqrt(pow((last_point_x - point_x), 2) + pow((last_point_y - point_y), 2)))
    if (point_x!=0 and point_y!=0 and dist>5):
        #if the point is accepted it is added to points list and its length added to list_len
        list_len.append(dist)
        length += dist
        last_point_x = point_x
        last_point_y = point_y
        points.append([point_x, point_y])
    #if length becomes greater then the expected length, removing points from the back to decrease length
    if (length>=snake_len):
        for i in range(len(list_len)):
            length -= list_len[0]
            list_len.pop(0)
            points.pop(0)
            if(length<=snake_len):
                break
    #initializing blank black image
    blank_img = np.zeros((480, 640, 3), np.uint8)
    #drawing the lines between all the points
    for i,j in enumerate(points):
        if (i==0):
            continue
        cv2.line(blank_img, (points[i-1][0], points[i-1][1]), (j[0], j[1]), (0, 0, 255), 5)
    cv2.circle(blank_img, (last_point_x, last_point_y), 5 , (10, 200, 150), -1)
    #if snake eats apple increase score and find new position for apple
    if  (last_point_x>random_x and last_point_x<(random_x+40) and last_point_y>random_y and last_point_y<(random_y+40)):
        score +=1
        random_x = random.randint(10, 550)
        random_y = random.randint(10, 400)
    #adding blank image to captured frame
    frame = cv2.add(frame,blank_img)
    #adding apple image to frame
    roi = frame[random_y:random_y+40, random_x:random_x+40]
    img_bg = cv2.bitwise_and(roi, roi, mask=apple_mask_inv)
    img_fg = cv2.bitwise_and(apple, apple, mask=apple_mask)
    dst = cv2.add(img_bg, img_fg)
    frame[random_y:random_y + 40, random_x:random_x + 40] = dst
    cv2.putText(frame, str("Score - "+str(score)), (250, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # checking for snake hitting itself
    if(len(points)>5):
        # a and b are the head points of snake and c,d are all other points
        b = points[len(points)-2]
        a = points[len(points)-1]
        for i in range(len(points)-3):
            c = points[i]
            d = points[i+1]
            if(intersect(a,b,c,d) and len(c)!=0 and len(d)!=0):
                temp = 0
                break
        if temp==0:
            break

    cv2.imshow("frame",frame)
    # increasing the length of snake 40px per second
    if((int(time())-start_time)>1):
        snake_len += 40
        start_time = int(time())
    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
cv2.putText(frame, str("Game Over!"), (100, 230), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
cv2.putText(frame, str("Press any key to Exit."), (180, 260), font, 1, (255, 200, 0), 2, cv2.LINE_AA)
cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()