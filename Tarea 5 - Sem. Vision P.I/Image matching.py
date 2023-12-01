# método knnMatch de la clase BFMatcher
import cv2 as cv
from matplotlib import pyplot as plt


#Load the images
image0 = cv.imread('image-matching/eynsham/p1.jpg', cv.IMREAD_GRAYSCALE)
image1 = cv.imread('image-matching/eynsham/p2.jpg', cv.IMREAD_GRAYSCALE)


#Perform ORB feature detection and description
orb = cv.ORB_create()
kp0, des0 = orb.detectAndCompute(image0, mask=None)
kp1, des1 = orb.detectAndCompute(image1, mask=None)

#Perform brute-force KNN matching
bf = cv.BFMatcher(normType=cv.NORM_HAMMING, crossCheck=False)
pairs_of_matches = bf.knnMatch(queryDescriptors=des0, trainDescriptors=des1, k=2)

#Sort the pairs of matches by distance
pairs_of_matches = sorted(pairs_of_matches, key=lambda x: x[0].distance)



#Apply the ratio test
matches = [x[0] for x in pairs_of_matches
           if len(x) > 1 and x[0].distance < 0.8 * x[1].distance]

#Draw the best 25 matches
img_matches = cv.drawMatches(img1=image0, keypoints1=kp0,
                             img2=image1, keypoints2=kp1,
                             matches1to2=matches[:25], outImg=None,
                             flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                             matchColor=(0, 255, 0))

cv.imwrite('orb-knn-matches.png', img_matches)

#Show the matches
plt.imshow(img_matches)
plt.show()