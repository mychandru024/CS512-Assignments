import cv2
import sys
import numpy as np
import os.path

#fileName = "Messi.jpg"
currChannel = 0
availableChannels = 3           
def slidehandler():
    global image
    n = 5
    kernel = np.ones((n,n),np.float32)/(n*n)
    dst = cv2.filter2D(src, -1, kernel)
    cv2.imshow(winName, image)

def process(fileName, image):
    keyPressed = input("Enter your choice: (press 'h' for help)")
    #image = cv2.imread(fileName)
    global currChannel
    global availableChannels
    if keyPressed == "i":
        #reload the image (cancel all previous processing)
        image = cv2.imread(fileName)
    elif keyPressed == "w":
        #save the current image (possibly processed)
        file = 'out.jpg'
        if os.path.isfile(file):
            os.remove(file)
        cv2.imwrite('out.jpg',image)
    elif keyPressed == "g":
        #convert image to gray scale using opencv conversion function
        #image = cv2.imread(fileName)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
    elif keyPressed == "G":
        #convert to gray scale using my own conversion
        pass
    elif keyPressed == "c":
        image = cv2.imread(fileName,currChannel)
        if currChannel % availableChannels == 0:
            currChannel = 0
        else:
            currChannel = currChannel + 1
        cv2.imshow('image',image)
        cv2.waitKey(0)
    elif keyPressed == "s":
        #convert image to grayscale and smooth using opencv function. Use track bar to control amount of amoothing
        win = "Grayed Image"
        image = cv2.imread(fileName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow(win, image)
        #cv2.waitKey(0)
        cv2.createTrackbar('s', win, 0, 255, slidehandler)
        cv2.waitKey(0)
    elif keyPressed == "S":
        #convert to gray scale and smooth using out own function
        pass
    elif keyPressed == "d":
        image = cv2.imread(fileName)
        print(image.shape)
        image = cv2.resize(image, (int(np.floor(image.shape[1]/2)), int(np.floor(image.shape[0]/2))))
        #print(image_DS.shape)
        #cv2.imshow('Under sampled by 2', image_DS)
        #cv2.waitKey(0)
    elif keyPressed == "D":
        image = cv2.imread(fileName)
        image = cv2.medianBlur(image, 5)
        #cv2.imshow("DS with smoothing", image)
        #cv2.waitKey(0)
    elif keyPressed == "x":
        image = cv2.imread(fileName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		dy, dx = np.gradient(image)
        #sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
        image = cv2.filter2D(image, image, sobelx)
        #kx = np.array(1000)
        #ky = np.array(1000)
        #cv2.getDerivKernels(1, 0, 1, True, "CV_32f")
        #print(kx)
        #print(ky)
    elif keyPressed == "y":
        image = cv2.imread(fileName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dy, dx = np.gradient(image)
        cv2.filter2D(image, image, dy)
        #kx = np.array(1000)
        #ky = np.array(1000)
        #kx, ky = cv2.getDerivKernels(0, 1, 1, True, CV_32f)
    elif keyPressed == "m":
        image = cv2.imread(fileName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif keyPressed == "r":
        image = cv2.imread(fileName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #angle = 45*np.pi/180
        rows = image.shape[0]
        cols = image.shape[1]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
        image = cv2.warpAffine(image, M, (cols, rows))
    elif keyPressed == "h":
        print("i: reload the image")
        print("w: save the image")
        print("g: convert the image to gray scale using opencv function")
        print("G: convert the image to gray scale using our own function")
        print("c: cycle through color channels")
        print("s: convert the image to gray scale and smooth it using opencv function, use trackbar")
        print("S: convert the image to gray scale and smooth it using our own function, use trackbar")
        print("d: Downsample by 2 without smoothing")
        print("D: Downsample by 2 with smoothing")
        print("x: convert to gray scale and convolve with x derivative filter")
        print("y: convert to gray scale and convolve with y derivative filter")
        print("m: show the magnitude of the gradient normalized to [0, 255]")
        print("p: convert to gray scale and plot the gradient vectors of the image every N pixels, use trackbar")
        print("r: convert to gray scale and rotate using an angle of theta degree")
        print("E: Exit the program")
        process(fileName, image)
    elif keyPressed == "E":
        sys.exit()
    else:
        print("Invalid option.. try again")
        process(fileName, image)
    process(fileName, image)
if len(sys.argv) > 1:
    #print(sys.argv[1])
    #read the image specified
    #image = cv2.imread(str(sys.argv[1]),cv2.IMREAD_UNCHANGED)
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    image = cv2.imread(str(sys.argv[1]))
    process(str(sys.argv[1]), image)
else:
    #capture an image and process it
    while True:
        cap = cv2.VideoCapture(0)
        if cap.isOpened() != True:
            cap.open()
        retval, image = cap.read()
        cv2.imwrite('Taken.jpg',image)
        #if retval:
            #print("Printing")
            #cv2.imshow('image',image)
            #cv2.waitKey(0)
        cap.release()
        image = cv2.imread('Taken.jpg')
        process('Taken.jpg', image)
    #cv2.destroyAllWindows()