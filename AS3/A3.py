import cv2
from matplotlib import pyplot as plt
import sys
import numpy as np
import numpy.linalg
#print(cv2.__version__)
"""
img = cv2.imread("image1.jpg")
#cv2.imwrite('image2.jpg',img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('image_gray.jpg',img_gray)
print("Original Image Shape : ", str(img.shape))
print("Grayed Image Shape : ", str(img_gray.shape))

print("Original Image size (number of pixels) : ", str(img.size))
print("Grayed Image size (number of pixels) : ", str(img_gray.size))

img_2 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE)
print("Grayed Image Shape : ", str(img_2.shape))
print("Grayed Image size (number of pixels) : ", str(img_2.size))

#cv2.imshow('Grayed Image', img_2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#plt.imshow(img, cmap = 'Reds')
#plt.show()

rows,cols = img_gray.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
img_rot = cv2.warpAffine(img_gray, M, (cols, rows))
cv2.imwrite('image_rotated.jpg',img_rot)
print("Rotated Image Shape : ", str(img_rot.shape))
print("Rotated Image size (number of pixels) : ", str(img_rot.size))

cv2.imshow('Rotated Image', img_rot)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

def readImage(fn):
	img_org = cv2.imread(fn)
	img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
	
	return img_gray

def detectCorners(img, ws, k, t):
	
	"""
	cv2.imshow('Image', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
	#sobel derivative of length 5
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	print(sobelx.shape)
	print(sobely.shape)
	
	#terms of the correlation matrix
	m0 = sobelx * sobelx
	m3 = sobely * sobely
	m1_m2 = sobelx * sobely
	
	#print(m0)
	
	correlation_matrix = np.matrix([[m0.sum(), m1_m2.sum()], [m1_m2.sum(), m3.sum()]])
	print(correlation_matrix)
	
	"""
	#det of a 2x2 matrix [[a, b], [c,d]] is ad-bc
	det = m0 * m3 - m1_m2 * m1_m2
	
	#trace of a square matrix is the sum of the elements along the diagonal
	trace = m0 + m3
	
	c_m = det - k * trace * trace
	"""
	
	h, w = img.shape
	
	dist = int(np.ceil(ws/2))
	
	img_org = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	corners = []
	#ignoring the boundaries and starting and ending at the positions such that we have prev and next neighbors
	for y in range(dist, h - dist):
		for x in range(dist, w - dist):
			#forming the window of gradients
			window_m0 = m0[y-dist:y+dist+1, x-dist:x+dist+1]
			window_m1_m2 = m1_m2[y-dist:y+dist+1, x-dist:x+dist+1]
			window_m3 = m3[y-dist:y+dist+1, x-dist:x+dist+1]
			
			#elements of the correlation matrix over this window
			sum_window_m0= window_m0.sum()
			sum_window_m1_m2 = window_m1_m2.sum()
			sum_window_m3 = window_m3.sum()
			
			#finding det and trace over this window
			det_window = (sum_window_m0 * sum_window_m3) - (sum_window_m1_m2 * sum_window_m1_m2)
			trace_window = sum_window_m0 + sum_window_m3
			
			r = det_window - k * (trace_window * trace_window)
		
			if r > t:
				#there is a corner in this window
				#assuming corner to be at the center of the window
				#corners.append([x, y, r])
				
				#corner localization
				win_pi = img[x-2:x+2, y-2:y+2]
				win_xgpi = sobelx[x-2:x+2, y-2:y+2]
				win_ygpi = sobely[x-2:x+2, y-2:y+2]
				print(img[0,0])
				v = [0 , 0]
				c = [[0, 0], [0, 0]]
				for i in range(5):
					gpi = [win_xgpi[i], win_ygpi[i]]
					gpiT = np.transpose(gpi)
					pi = win_pi[x+i, y+i]
					#print(gpi)
					#print(gpiT)
					print(win_pi[x+i, y+i])
					t1 = np.matmul(gpi, gpiT)
					c[0][0] = c[0][0] + t1[0][0]
					c[0][1] = c[0][1] + t1[0][1]
					c[1][0] = c[1][0] + t1[1][0]
					c[1][1] = c[1][1] + t1[1][1]
					
					t2 = np.matmul(t1 , pi)
					v[0] = v[0] + temp[0]
					v[1] = v[1] + temp[1]
				p = np.matmul(np.linalg.inv(c), v)
				corners.append([p[0], p[1], r])
				img_org.itemset((p[1], p[0], 0 ), 255)
				img_org.itemset((p[1], p[0], 1 ), 0)
				img_org.itemset((p[1], p[0], 2 ), 0)
	print("Number of corners: ", len(corners))
	return img_org, corners

def main():
	if len(sys.argv) > 1:
		#print(len(sys.argv))
		#print(sys.argv[0])
		fileName = str(sys.argv[1])
		window_size = int(sys.argv[2])
		k = float(sys.argv[3])
		threshold = int(sys.argv[4])
	else:
		fileName = input("Image path : ")
		window_size = input("Window size: ")
		k = input("weight of trace : ")
		threshold = input("Threshold to detect corner : ")
	
	img = readImage(fileName)
	img_with_corners1, corners1 = detectCorners(img, window_size, k, threshold)
	
	img2 = cv2.imread('image_rotated.jpg', cv2.IMREAD_GRAYSCALE)
	img_with_corners2, corners2 = detectCorners(img2, window_size, k, threshold)
	
	#sort corners in both images by their r values
	corners1 = sorted(corners1, key=lambda tup: tup[2])
	corners2 = sorted(corners2, key=lambda tup: tup[2])
	
	"""
	cv2.imshow('Image', img_with_corners2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	"""
	
	#img1 = cv2.imread('image1.jpg',0)
	#img2 = cv2.imread('image_rotated.jpg',0) 

	# Initiate SIFT detector
	orb = cv2.ORB_create()

	# find the keypoints and descriptors with SIFT
	#kp1, des1 = orb.detectAndCompute(img1,None) # returns keypoints and descriptors
	#kp2, des2 = orb.detectAndCompute(img2,None)
	
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	#match 30 corners with highest r values
	matches = bf.match(corners1[:30],corners2[:30])

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	"""
	draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
	"""
	# Draw first 10 matches.
	img3 = cv2.drawMatches(img_with_corners1,corners1,img_with_corners2,corners2,matches[:10], None, flags=2)

	plt.imshow(img3),plt.show()
	


if __name__ == "__main__":
	main()



