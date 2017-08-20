import cv2
import numpy as np
from time import time
import random
import math
#initializing font
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
#function for detecting blue color
def detect_blue(hsv):
    lower = np.array([110, 50, 50])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, kernel_erode, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def onSegment(p,q,r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True

    return False

def orientation(p,q,r):
    val = int(((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1])))
    if val == 0:
        return 0
    elif (val>0):
        #clockwise
        return 1
    else:
        return 2

def intersect(p,q,r,s):
    o1 = orientation(p, q, r)
    o2 = orientation(p, q, s)
    o3 = orientation(r, s, p)
    o4 = orientation(r, s, q)
    if(o1 != o2 and o3 != o4):
        return True
    """if (o1 == 0 and onSegment(p, r, q)):
        return True

    if (o2 == 0 and onSegment(p, s, q)):
        return True

    if (o3 == 0 and onSegment(r, p, s)):
        return True

    if (o4 == 0 and onSegment(r, q, s)):
        return True
"""
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
a,b,c,d = [],[],[],[]
while 1:
    xr, yr, wr, hr = 0, 0, 0, 0
    #xb, yb, wb, hb = 0, 0, 0, 0
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
        list_len.append(dist)
        length += dist
        last_point_x = point_x
        last_point_y = point_y
        points.append([point_x, point_y])

    #print(blank_img[point_y][point_x][2])
    #if blank_img[point_y][point_x][2]==0:
    #    blank_img[point_y][point_x] = [0,0,255]
    if (length>=snake_len):
        for i in range(len(list_len)):
            length -= list_len[0]
            list_len.pop(0)
            points.pop(0)
            if(length<=snake_len):
                break

    blank_img = np.zeros((480, 640, 3), np.uint8)
    for i,j in enumerate(points):
        if (i==0):
            continue
        cv2.line(blank_img, (points[i-1][0], points[i-1][1]), (j[0], j[1]), (0, 0, 255), 5)
        #cv2.circle(blank_img, (points[i-1][0], points[i-1][1]), 2, (0, 0, 0), -1)
    #cv2.circle(blank_img, (last_point_x, last_point_y), 2 , (255, 0, 0), -1)
    if  (last_point_x>random_x and last_point_x<(random_x+40) and last_point_y>random_y and last_point_y<(random_y+40)):
        score +=1
        random_x = random.randint(10, 550)
        random_y = random.randint(10, 400)
    #print(blank_img[last_point_y,last_point_x],"Break")
    frame = cv2.add(frame,blank_img)
    #cv2.imshow("new mask",maskred)
    #adding apple image to frame
    roi = frame[random_y:random_y+40, random_x:random_x+40]
    img_bg = cv2.bitwise_and(roi, roi, mask=apple_mask_inv)
    img_fg = cv2.bitwise_and(apple, apple, mask=apple_mask)
    dst = cv2.add(img_bg, img_fg)
    frame[random_y:random_y + 40, random_x:random_x + 40] = dst
    cv2.putText(frame, str("Score - "+str(score)), (250, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if(len(points)>5):
        b = points[len(points)-2]
        a = points[len(points)-1]
        for i in range(len(points)-3):
            c = points[i]
            d = points[i+1]
            if(intersect(a,b,c,d) and len(c)!=0 and len(d)!=0):
                print(intersect(a,b,c,d),a,b,c,d)
                temp = 0
                break
        if temp==0:
            break



    cv2.imshow("frame",frame)
    #cv2.imshow("img",blank_img)
    #blank_img_hsv = cv2.cvtColor(blank_img, cv2.COLOR_BGR2HSV)
    #maskred_snake = detect_red(blank_img_hsv)
    #maskblue_snake = detect_blue(blank_img_hsv)
    #part_blue = np.argwhere(blank_img == (255, 0, 0))[:, 0:2]
    #part_red = np.argwhere(blank_img == (0, 0, 255))[:, 0:2]
    """if(maskblue_snake in maskred_snake and last_point_y!=0 and last_point_x!=0):
        print(last_point_x,last_point_y)
        print("game over")
        break
    cv2.imshow("red",maskred_snake)
    cv2.imshow("blue",maskblue_snake)"""
    if((int(time())-start_time)>1):
        snake_len += 40
        start_time = int(time())
    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
cv2.putText(frame, str("Game Over!"), (100, 230), font, 3, (255, 0, 0), 3, cv2.LINE_AA)
cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()