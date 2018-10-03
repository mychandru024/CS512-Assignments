import numpy as np
import cv2
import pickle
import sys
from matplotlib import pyplot as plt

lpoints = []
rpoints = []

n_lpoints = []
n_rpoints = []

left_point_selected = False

def find_epipolar_line(p, F, leftPoint = True):
 if leftPoint:
  coefs = np.matmul(F, [ p[0], p[1], 1 ])
 else:
  coefs = np.matmul(np.transpose(F), [ p[0], p[1], 1 ])
 return coefs

def find_epipoles(F):
 U, D, Vt = np.linalg.svd(np.transpose(F), full_matrices=True)
 #right_epipole = [ U[0,2]/U[2,2], U[1,2]/U[2,2] ]
 right_epipole = [ U[2,0]/U[2,2], U[2,1]/U[2,2] ]
 V = np.transpose(Vt)
 #left_epipole = [ V[0,2]/V[2,2], V[1,2]/V[2,2] ]
 left_epipole = [ V[2,0]/V[2,2], V[2,1]/V[2,2] ]
 return left_epipole, right_epipole

def normalize(value, mean, sd):
 return (value - mean) / sd

def cal_mean(p):
 return np.mean(p)

def cal_SD(p):
 return np.std(p)

def find_fundamental_matrix():
 global lpoints, rpoints, n_lpoints, n_rpoints
 #lx_mean = 0, ly_mean = 0, rx_mean, ry_mean, lx_sd = 0, ly_sd = 0, rx_sd = 0, ry_sd = 0
 lx = []
 ly = []
 rx = []
 ry = []

 for point in lpoints:
  lx.append(point[0])
  ly.append(point[1])

 for point in rpoints:
  rx.append(point[0])
  ry.append(point[1])

 lx_mean = cal_mean(lx)
 ly_mean = cal_mean(ly)
 rx_mean = cal_mean(rx)
 ry_mean = cal_mean(ry)
 lx_sd = cal_SD(lx)
 ly_sd = cal_SD(ly)
 rx_sd = cal_SD(rx)
 ry_sd = cal_SD(ry)

 for point in lpoints:
  n_x = normalize(point[0], lx_mean, lx_sd)
  n_y = normalize(point[1], ly_mean, ly_sd)
  n_lpoints.append((n_x,n_y))

 for point in rpoints:
  n_x = normalize(point[0], rx_mean, rx_sd)
  n_y = normalize(point[1], ry_mean, ry_sd)
  n_rpoints.append((n_x,n_y))

 """
 print("Original points from left image: ", lpoints,"\n")
 print("Normalized points from left image: ", n_lpoints,"\n")
 print("Original points from right image: ", rpoints,"\n")
 print("Normalized points from left image: ", n_rpoints,"\n")
 """

 M = []
 for i in range(len(lpoints)):
  left_point = tuple(n_lpoints[i])
  right_point = tuple(n_rpoints[i])
  M.append( [ 
             right_point[0] * left_point[0], left_point[1] * right_point[0], right_point[0], \
             right_point[1] * left_point[0], right_point[1] * left_point[1], right_point[1], \
             left_point[0], left_point[1], 1 
            ] )
  #M.append(row)
 #print(len(M))
 
 U, D, Vt = np.linalg.svd(M, full_matrices=True) 
 #V = np.transpose(Vt)
 #print(V.shape)
 est_F = np.split(Vt[-1], 3)
 #print(est_F)

 u, d, vt = np.linalg.svd(est_F, full_matrices=True)
 d_new = [[d[0], 0, 0], [0, d[1], 0], [0, 0, 0]]

 est_f_2 = np.matmul( np.matmul(u,d_new), vt)
 #print(est_f_2)

 return est_f_2

def click_left(event, x, y, flags, param):
 global lpoints, left_point_selected
 if event == cv2.EVENT_LBUTTONUP:
  if not left_point_selected:
   lpoints.append((x, y))
   left_point_selected = True
   pickle.dump(lpoints, open("../data/left_points.pkl", "wb"))

def click_right(event, x, y, flags, param):
 global rpoints, left_point_selected
 if event == cv2.EVENT_LBUTTONUP:
  if left_point_selected == True:
   rpoints.append((x, y))
   left_point_selected = False
   pickle.dump(rpoints, open("../data/right_points.pkl", "wb"))

if __name__ == "__main__":
 if len(sys.argv) > 1:
  option = int(sys.argv[1])
  if option == 1:
   img1 = cv2.imread("../data/rock-l.tif")
   img2 = cv2.imread("../data/rock-r.tif")

   img1_g = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
   img2_g = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
   X1, Y1 = img1_g.shape
   X2, Y2 = img2_g.shape
   cv2.imshow("Stereo Vision : left", img1_g)
   cv2.imshow("Stereo Vision : right", img2_g)

   cv2.setMouseCallback("Stereo Vision : left", click_left)
   cv2.setMouseCallback("Stereo Vision : right", click_right)
   cv2.waitKey(0) & F
  elif option ==2:
   lpoints = pickle.load(open("../data/left_points.pkl", "rb"))
   rpoints = pickle.load(open("../data/right_points.pkl", "rb"))
   
   """
   img1 = cv2.imread("../data/rock-l.tif")
   img2 = cv2.imread("../data/rock-r.tif")

   img1_g = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
   img2_g = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
 
   bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

   # Match descriptors.
   matches = bf.match(lpoints, rpoints)

   # Sort them in the order of their distance.
   matches = sorted(matches, key=lambda x: x.distance)

   # Draw first 10 matches.
   img3 = cv2.drawMatches(img1_g, lpoints, img2_g, rpoints, matches[:10], None, flags=2)
   #cv2.imwrite('interest_points_matched_in_captured_and_transformed.jpg', img3)
   plt.imshow(img3), plt.show()
   """

   F = find_fundamental_matrix()
   le, re = find_epipoles(F)
   print("Left epipole: ", le,"\n")
   print("Right epipole: ",re,"\n")

   coef = find_epipolar_line(n_lpoints[0], F, True)
   print("Coefficients of the right epipolar line corresponding to point {0} in left image is: {1}\n".format(lpoints[0], coef))

   coef = find_epipolar_line(n_rpoints[0], F, False)
   print("Coefficients of the left epipolar line corresponding to point {0} in right image is: {1}\n".format(rpoints[0], coef))
 else:
  print("\nFirst execute the program with commandline argument 1 to select the point correspondence\nThen re-execute the program with commandline arguement 2 to see the result of processing.\n")

