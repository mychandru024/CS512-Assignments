import cv2
#print(cv2.__version__)

import sys
print(len(sys.argv))
if len(sys.argv) > 1:
 print(sys.argv[1])
 #read the image specified
 img = cv2.imread('image.jpg',cv2.IMREAD_UNCHANGED)
 cv2.imshow('image',img)
 cv2.waitKey(10) & 255 
 cv2.destroyAllWindows()
else:
 #capture an image and process it
 print("To Do")
