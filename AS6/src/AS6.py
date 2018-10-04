import cv2
import numpy as np
import sys

output_folder = "../data/output/"

"""
import pyglet
from pyglet.window import key

window = pyglet.window.Window()

@window.event
def on_key_press(symbol, modifiers):
 if symbol == key.P:
  print('The "P" key was pressed.')
"""
if __name__ == "__main__":
 #pyglet.app.run()
 threshold = 0.7
 file_name = "../data/UGS06_001.mpg"
 if len(sys.argv) > 2:
  file_name = str(sys.argv[1])
  threshold = float(sys.argv[2])
 print("Data file: ", file_name)
 print("Threshold for reliability: ", threshold)
 cap = cv2.VideoCapture(file_name)
 ret, frame = cap.read()
 i = 1
 
 while i < 27:
  ret, frame = cap.read()
  i = i + 1
 
 while ret:
  
  frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #print(frame[0,0])
  
  """
  cv2.imshow("First Frame", frame_gray)
  key = cv2.waitKey(0)# & 0xff
  """
  
  sobelx = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=5)
  sobely = cv2.Sobel(frame_gray,cv2.CV_64F,0,1,ksize=5)
  
  """
  cv2.imshow("sobelx", sobelx)
  cv2.imshow("sobely", sobely)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  """
  
  ret, frame_next = cap.read()
  frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
  if ret:
   #calculating frame difference, using forward difference
   frame_diff = np.zeros((frame_gray.shape[0], frame_gray.shape[1]))
   for x in range(2, frame_gray.shape[0] - 1):
    for y in range(2, frame_gray.shape[1] - 1):
     frame_diff[x][y] = frame_gray[x][y] - frame_next_gray[x][y]
  
  #A = np.zeros((6,6))
  #B = np.zeros((6,1))

  for x in range(2, frame_gray.shape[0] - 3):
   for y in range(2, frame_gray.shape[1] - 3):
    sum_IxIy = sum_IxIy_x2 = sum_IxIy_x = sum_IxIy_y = sum_IxIy_y2 = sum_IxIy_xy = sum_Ix2 = sum_Ix2_x = sum_Ix2_xy = 0
    sum_Ix2_x2 = sum_Ix2_y = sum_Ix2_y2 = sum_Iy2 = sum_Iy2_x = sum_Iy2_x2 = sum_Iy2_y = sum_Iy2_y2 = sum_Iy2_xy = 0 
    sum_IxIt = sum_IxIt_x = sum_IxIt_y = sum_IyIt = sum_IyIt_x = sum_IyIt_y = 0
    """
    sum_IxIy_x2 = 0, sum_IxIy_x = 0, sum_IxIy_y = 0, sum_IxIy_y2 = 0, sum_IxIy_xy = 0, sum_Ix2 = 0, sum_Ix2_x = 0, sum_Ix2_xy = 0
    sum_Ix2_x2 = 0, sum_Ix2_y = 0, sum_Ix2_y2 = 0, sum_Iy2 = 0, sum_Iy2_x = 0, sum_Iy2_x2 = 0, sum_Iy2_y = 0, sum_Iy2_y2 = 0, sum_Iy2_xy = 0 
    sum_IxIt = 0, sum_IxIt_x = 0, sum_IxIt_y = 0, sum_IxIt = 0, sum_IxIt_x = 0, sum_IxIt_y = 0
    """
    for m in range(-2, 3):
     for n in range(-2, 3):
      sum_IxIy += sobelx[x+m,y+n]*sobely[x+m,y+n]
      sum_IxIy_x2 += sobelx[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][0]*frame[x+m,y+n][0]
      sum_IxIy_x += sobelx[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][0]
      sum_IxIy_y += sobelx[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][1]
      sum_IxIy_y2 += sobelx[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][1]*frame[x+m,y+n][1]
      sum_IxIy_xy += sobelx[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][0]*frame[x+m,y+n][1]
      sum_Ix2 += sobelx[x+m,y+n]*sobelx[x+m,y+n]
      sum_Ix2_x += sobelx[x+m,y+n]*sobelx[x+m,y+n]*frame[x+m,y+n][0]
      sum_Ix2_xy += sobelx[x+m,y+n]*sobelx[x+m,y+n]*frame[x+m,y+n][0]*frame[x+m,y+n][1]
      sum_Ix2_x2 += sobelx[x+m,y+n]*sobelx[x+m,y+n]*frame[x+m,y+n][0]*frame[x+m,y+n][0]
      sum_Ix2_y += sobelx[x+m,y+n]*sobelx[x+m,y+n]*frame[x+m,y+n][1]
      sum_Ix2_y2 += sobelx[x+m,y+n]*sobelx[x+m,y+n]*frame[x+m,y+n][1]*frame[x+m,y+n][1]
	  
      sum_Iy2 += sobely[x+m,y+n]*sobely[x+m,y+n]
      sum_Iy2_x += sobely[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][0]
      sum_Iy2_x2 += sobely[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][0]*frame[x+m,y+n][0]
      sum_Iy2_y += sobely[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][1]
      sum_Iy2_y2 += sobely[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][1]*frame[x+m,y+n][1]
      sum_Iy2_xy += sobely[x+m,y+n]*sobely[x+m,y+n]*frame[x+m,y+n][0]*frame[x+m,y+n][1]
	  
      sum_IxIt += sobelx[x+m,y+n]*frame_diff[x+m,y+n]
      sum_IxIt_x += sobelx[x+m,y+n]*frame_diff[x+m,y+n]*frame[x+m,y+n][0]
      sum_IxIt_y += sobelx[x+m,y+n]*frame_diff[x+m,y+n]*frame[x+m,y+n][1]
      sum_IyIt += sobely[x+m,y+n]*frame_diff[x+m,y+n]
      sum_IyIt_x += sobely[x+m,y+n]*frame_diff[x+m,y+n]*frame[x+m,y+n][0]
      sum_IyIt_y += sobely[x+m,y+n]*frame_diff[x+m,y+n]*frame[x+m,y+n][1]

    A = [ [sum_Ix2, sum_Ix2_x, sum_Ix2_y, sum_IxIy, sum_IxIy_x, sum_IxIy_y], \
		  [sum_Ix2_x, sum_Ix2_x2, sum_Ix2_xy, sum_IxIy_x, sum_IxIy_x2, sum_IxIy_xy], \
		  [sum_Ix2_y, sum_Ix2_xy, sum_Ix2_y2, sum_IxIy_y, sum_IxIy_xy, sum_IxIy_y2], \
		  [sum_IxIy, sum_IxIy_x, sum_IxIy_y, sum_Iy2, sum_Iy2_x, sum_Iy2_y], \
		  [sum_IxIy_x, sum_IxIy_x2, sum_IxIy_xy, sum_Iy2_x, sum_Iy2_x2, sum_Iy2_xy], \
		  [sum_IxIy_y, sum_IxIy_xy, sum_IxIy_y2, sum_Iy2_y, sum_Iy2_xy, sum_Iy2_y2] 
		]
    B = [-sum_IxIt, -sum_IxIt_x, -sum_IxIt_y, -sum_IyIt, -sum_IyIt_x, -sum_IxIy_y]
	
    T = np.matmul(np.transpose(A), A)
	
    e = np.linalg.eigvals(T)
    #print(e)
	
    e = np.sort(e)
    #print("Eigen values: ", e)
	
    reliability = e[4]/e[5]
	
    if reliability < threshold:
     continue
	
    try:
     params = np.matmul(np.linalg.inv(A), B)
    except np.linalg.linalg.LinAlgError as lae:
     continue
    #print("Parameters: ", params)
	
    V = [params[0] + params[1] * frame[x+m,y+n][0] + params[2] * frame[x+m,y+n][1], params[3] + params[4] * frame[x+m,y+n][0] + params[5] * frame[x+m,y+n][1]]
    #print("Optical Flow vector: ", V)
	
    p = 5
    q = 5
	
    if V[0] < 0:
     p = -5
	
    if V[1] < 0:
     q = -5
	
    if reliability > 0.7 and reliability < 0.8:
     cv2.arrowedLine(frame, (x, y), (x + p, y + q), (0, 50, 0),1)
    elif reliability > 0.8 and reliability < 0.9:
     cv2.arrowedLine(frame, (x, y), (x + p, y + q), (0, 100, 0),2)
    elif reliability > 0.9:
     cv2.arrowedLine(frame, (x, y), (x + p, y + q), (0, 255, 0),3)

    """
    f = 'plot' + str(i)+ '.png'
    cv2.imwrite(f, frame)
    i = i + 1
    """
    #frame = frame_next
	
    """
    cv2.imshow(f, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
	  
    #break
   #break
  f = output_folder + 'frame_' + str(i)+ '.png'
  cv2.imwrite(f, frame)
  i = i + 1
  frame = frame_next
 
 
 
 