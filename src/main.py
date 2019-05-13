"""
main.py
Opens a video file, tracks object motion in it,
and generates text in the wake of the moving object(s)
Adapted from the example: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html

Zena Abulhab & Melody Mao
CS365 Spring 2019
Project 6
"""

import sys
import cv2 as cv
import numpy as np
import random

green = (0, 255, 0)

'''
Returns the biggest contour in the given hierarchy that contains the given point
'''
# def getContourFromInteriorPoint(point, contours, hierarchy):
# 	largestContourIDX = 0
# 	largestContourArea = 0
# 	for i in range( len( contours ) ):
# 		inContour = cv.pointPolygonTest( contours[i], point, False )  # False => don't measure distance to edge
#
# 		thisContourArea = cv.contourArea( contours[i] )
# 		if inContour > 0 and thisContourArea > largestContourArea:
# 			print("aaaah", inContour)
# 			largestContourIDX = i
# 			largestContourArea = thisContourArea
#
# 	if largestContourArea != 0:
# 		return contours[largestContourIDX]
# 	else:
# 		return None
	
	# print("hierarchy", len(hierarchy))
	#
	# currentContour = None # current best match for contour containing this point
	# for idx in range(hierarchy.shape[1]):
	# 	c = contours[idx]
	# 	parent = hierarchy[0][idx][2]
	#
	# 	# -1 = outside contour, 0 = on contour, 1 = in contour
	# 	contourTest = cv.pointPolygonTest( c, point, False )  # False => don't measure distance to edge
	# 	if contourTest != -1: # if point is in/on contour
	# 		print("HERE I AM")
	# 		if parent == -1: # no parent
	# 			currentContour = c
	# 		else:
	# 			currentContour = contours[parent]
	#
	# return currentContour

'''
Mouse click callback function for initial frame window
'''
def onMouseClick(event, x, y, flags, param):
	# clickPts = param["pointList"]
	frame = param["frame"]
	# contours = param["contours"]
	# hierarchy = param["hierarchy"]
	boxPts = param["boxPts"]
	
	if event == cv.EVENT_LBUTTONDOWN: # drag start
		box = {"x1": int(x), "y1": int(y)}
		boxPts.append( box )
		# cv.circle(frame, pt, 2, green)
		#
		# # get contour and draw it
		# contour = getContourFromInteriorPoint(pt, contours, hierarchy)
		# print(contour)
		# cv.drawContours(frame, [contour], 0, green)
	elif event == cv.EVENT_LBUTTONUP: # drag end
		thisBox = boxPts[-1]
		thisBox["x2"] = int(x)
		thisBox["y2"] = int(y)
		
		# draw box
		cv.line( frame, (thisBox["x1"], thisBox["y1"]),
				 (thisBox["x1"], thisBox["y2"]), green, thickness=2 )
		cv.line( frame, (thisBox["x1"], thisBox["y1"]),
				 (thisBox["x2"], thisBox["y1"]), green, thickness=2 )
		cv.line( frame, (thisBox["x1"], thisBox["y2"]),
				 (thisBox["x2"], thisBox["y2"]), green, thickness=2 )
		cv.line( frame, (thisBox["x2"], thisBox["y1"]),
				 (thisBox["x2"], thisBox["y2"]), green, thickness=2 )
		

#TODO: should this be for selecting multiple or just one?
def selectObjects(frame):
	frameCopy = frame.copy()
	# grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	# threshold = 127
	# maxVal = 255
	# #ret, thresholdFrame = cv.threshold(grayFrame, threshold, maxVal, cv.THRESH_BINARY)
	# blockSize = 11
	# c = 2 # "just a constant subtracted from the mean"
	# thresholdFrame = cv.adaptiveThreshold(grayFrame, maxVal, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
	# 									  cv.THRESH_BINARY, blockSize, c)
	# #											src				retrieval mode, approximation method
	# im2, contours, hierarchy = cv.findContours( thresholdFrame, cv.RETR_TREE,   cv.CHAIN_APPROX_SIMPLE )
	# print("aaaaa", type(hierarchy), hierarchy.shape)
	# print("bbbbb", type(contours), len(contours))
	# #cv.imshow("threshold", thresholdFrame)
	#
	# frameCopy2 = frame.copy()
	# frameCopy2 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	# frameCopy2[ frameCopy >= 170 ] = 255
	# frameCopy2[ frameCopy < 170 ] = 127
	# frameCopy2[ frameCopy < 85 ] = 0
	# #cv.imshow("segmented", frameCopy2)
	
	#TODO: aaaaaaa
	# unique = []
	# for row in range(frameCopy2.shape[0]):
	# 	unique.append( np.unique(frameCopy2[row], axis=0) )
	# unique, mapping = numpy.unique( numpy.array( labels ), return_inverse=True )
	
	# colorVals = [0, 127, 255]
	# allContours = []
	# for i in range(3):
	# 	for j in range(3):
	# 		for k in range(3):
	# 			curColor = (colorVals[i], colorVals[j], colorVals[k])
	#
	# 			mask = cv.inRange( frameCopy2, curColor, curColor )
	# 			# print(type(mask[0][0]))
	# 			# result = cv.bitwise_and( frameCopy2, frameCopy2, mask=mask )
	# 			# title = "" + str(i) + ", " + str(j) + ", " + str(k)
	# 			# cv.imshow(title, result)
	#
	# 			im2, contours, hierarchy = cv.findContours( mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE )
	# 			#print("length of one", len(contours))
	# 			#allContours.extend( contours ) # get all of hem
	# 			for c in contours:
	# 				thisContourArea = cv.contourArea( c )
	# 				if thisContourArea > 150:
	# 					allContours.append(c)
	
		
	cv.namedWindow( "initial frame" )
	# clickCoords = []
	boxPts = []
	param = { "boxPts": boxPts, "frame": frameCopy }
	# param = { "pointList": clickCoords, "frame": frameCopy, "contours": allContours, "hierarchy": hierarchy }
	cv.setMouseCallback( "initial frame", onMouseClick, param )
	
	# print("??")
	# print(len(allContours))
	#
	# for i in range(len(allContours)):
	# 	cv.drawContours(frameCopy, allContours, i, (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)))
	# cv.imshow("initial frame", frameCopy)
	
	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv.imshow( "initial frame", frameCopy )
		key = cv.waitKey( 1 ) & 0xFF
		
		# if the 'q' key is pressed, break from the loop
		if key == ord( "q" ):
			break
	
	# close window
	cv.destroyWindow("initial frame")
	
	'''
	get all contours in image as in OR system
	store user click point(s)
	test against all contours w/ pointPolygonTest to figure out which one it's in
	draw in selected contour
	return selected contour/list of selected contours?
	'''
	
	return boxPts

def main(argv):
	# check for command-line argument
	if len(argv) < 2:
		print("Usage: python3 main.py [video filename]")
		exit()
	
	cap = cv.VideoCapture(argv[1])
	
	# check to make sure video opened successfully
	if not cap.isOpened():
		print( "Error opening video stream or file" )
		exit(-1)
	
	# read in first frame
	ret, frame1 = cap.read()
	prevGray = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
	
	boxPts = selectObjects(frame1)
	input("AHA") # temp just for pausing
	
	# initialize array for HSV representation of flow blobs
	hsvBlobs = np.zeros_like(frame1)
	hsvBlobs[...,1] = 255 # give all pixels 100% saturation
	
	# loop through frames until video is completed
	while cap.isOpened():
		ret, frame2 = cap.read()
		
		# stop immediately on a non-existent frame
		if frame2 is None:
			break
		
		nextGray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
		flow = cv.calcOpticalFlowFarneback(prevGray, nextGray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		
		#TODO: aaaaaaa just added
		for box in boxPts:
			# get average flow across object bbox in x and y directions
			print(box)
			print(flow.shape)
			print("YEESH", flow[box["y1"]:box["y2"], box["x1"]:box["x2"], 0].shape)
			print()
			boxAvgXFlow = np.mean( flow[box["y1"]:box["y2"], box["x1"]:box["x2"], 0] )
			boxAvgYFlow = np.mean( flow[box["y1"]:box["y2"], box["x1"]:box["x2"], 1] )
			print("average flow:", boxAvgXFlow, boxAvgYFlow)
			
			# move bbox
			box["x1"] += int(boxAvgXFlow)
			box["x2"] += int(boxAvgXFlow)
			box["y1"] += int(boxAvgYFlow)
			box["y2"] += int(boxAvgYFlow)
		
			# draw moved box into blob image
			print(type(box["y1"]), box["y1"])
			for i in range(box["y1"], box["y2"]):
				for j in range(box["x1"], box["x2"]):
					hsvBlobs[i, j, 0] = 255
					hsvBlobs[i, j, 2] = 255
		
		# #TODO: make trails & remove confetti
		# # get flow vector info to turn into HSV color info
		# mag, ang = cv.cartToPolar(flow[...,0], flow[...,1]) # output: vector magnitudes, angles
		#
		# # if flow is big enough
		# # (ignores frames w/ almost no motion where very small flow values get normalized way too high,
		# # resulting in frames w/ confetti)
		# print("mag max", mag.max())
		# if mag.max() > 50.:
		# 	# vector angle => hue
		# 	angDegrees = ang*180/np.pi/2 # convert radians to degrees
		# 	hsvBlobs[...,0] = angDegrees
		#
		# 	# vector magnitude => value
		# 	magNormalized = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
		# 	magThreshold = 5.
		# 	mask = magNormalized > magThreshold # at each index, boolean value for whether > threshold
		#
		# 	# uncomment for displayable version of mask
		# 	mask = mask.astype(np.float)
		# 	maskNorm = mask * 255
		# 	cv.imshow( 'blob mask', maskNorm )
		#
		# 	for i in range(mask.shape[0]):
		# 		for j in range(mask.shape[1]):
		# 			if mask[i][j]:
		# 				hsvBlobs[i,j,2] = magNormalized[i,j]

		bgrBlobs = cv.cvtColor(hsvBlobs, cv.COLOR_HSV2BGR)
		#cv.imshow( 'BLOBS', bgrBlobs ) # uncomment to display just blobs
		display = cv.add(frame2, bgrBlobs) # draw flow blobs over original frame
		cv.imshow('optical flow', display)
		
		#k = cv.waitKey(1000) & 0xff
		k = cv.waitKey(30) & 0xff
		# if k == 27:
		# 	break
		# elif k == ord('s'): # save frames on key press
		# 	cv.imwrite('opticalfb.png', frame2)
		# 	cv.imwrite('opticalhsv.png', bgr)
		
		prevGray = nextGray
	
	# clean up
	cap.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	main(sys.argv)