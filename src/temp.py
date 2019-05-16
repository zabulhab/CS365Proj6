"""
temp.py
Opens a video file, attempts to track object motion in it,
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
def getContourFromInteriorPoint(point, contours, hierarchy):
	# largestContourIDX = 0
	# largestContourArea = 0
	# for i in range( len( contours ) ):
	# 	inContour = cv.pointPolygonTest( contours[i], point, False )  # False => don't measure distance to edge
	#
	# 	thisContourArea = cv.contourArea( contours[i] )
	# 	if inContour > 0 and thisContourArea > largestContourArea:
	# 		print("aaaah", inContour)
	# 		largestContourIDX = i
	# 		largestContourArea = thisContourArea
	#
	# if largestContourArea != 0:
	# 	return contours[largestContourIDX]
	# else:
	# 	return None

	print("hierarchy", len(hierarchy))
	
	currentContour = None # current best match for contour containing this point
	for idx in range(hierarchy.shape[1]):
		c = contours[idx]
		parent = hierarchy[0][idx][2]
	
		# -1 = outside contour, 0 = on contour, 1 = in contour
		contourTest = cv.pointPolygonTest( c, point, False )  # False => don't measure distance to edge
		if contourTest != -1: # if point is in/on contour
			print("HERE I AM")
			if parent == -1: # no parent
				currentContour = c
			else:
				currentContour = contours[parent]
	
	return currentContour

'''
Mouse click callback function for initial frame window
'''
def onMouseClick( event, x, y, flags, param ):
	clickPts = param["pointList"]
	frame = param[ "frame" ]
	# contours = param["contours"]
	# hierarchy = param["hierarchy"]
	
	if event == cv.EVENT_LBUTTONDOWN: # left mouse click
		pt = (x, y)
		clickPts.append(pt)
		cv.circle(frame, pt, 2, green)
	
		# # get contour and draw it
		# contour = getContourFromInteriorPoint(pt, contours, hierarchy)
		# contours.append( contour )
		# # print(contour)
		# cv.drawContours(frame, [contour], 0, green)


# TODO: should this be for selecting multiple or just one?
def selectObjects( frame, blobs ):
	frameCopy = frame.copy( )
	
	bgrBlobs = cv.cvtColor( blobs, cv.COLOR_HSV2BGR )
	cv.imshow( "blobs", bgrBlobs )
	
	# maxColorVal = blobs.max()
	# print("max color", maxColorVal)
	# posterizeBoundary = 20
	# blobs[ blobs >= posterizeBoundary ] = 255
	# blobs[ blobs < posterizeBoundary ] = 0
	#cv.imshow("blobs", blobs)
	# cv.waitKey(0)
	# input("aaa")
	
	# maxVal = 255
	# blockSize = 11
	# thresholdFrame = cv.adaptiveThreshold(grayFrame, maxVal, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
	# 									  cv.THRESH_BINARY, blockSize, c)
	# im2, contours, hierarchy = cv.findContours( thresholdFrame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE )
	
	cv.namedWindow( "initial frame" )
	clickCoords = []
	# boxPts = [ ]
	# param = { "boxPts": boxPts, "frame": frameCopy }
	param = { "pointList": clickCoords, "frame": frameCopy }
	cv.setMouseCallback( "initial frame", onMouseClick, param )
	
	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv.imshow( "initial frame", frameCopy )
		key = cv.waitKey( 1 ) & 0xFF
		
		# if the 'q' key is pressed, break from the loop
		if key == ord( "q" ):
			break
	
	click1 = clickCoords[0]
	# print("click", click1)
	# blobColor = blobs[click1[0], click1[1]]
	# print("blob color", blobColor)
	# maxDiff = 100.
	# darkest = (blobColor[0] - 10., blobColor[1] - 10., blobColor[2] - 15.)
	# lightest = (blobColor[0] + 10., blobColor[1] + 10., blobColor[2] + 15.)
	# print("between", darkest, "and", lightest)
	# mask = cv.inRange( blobs, darkest, lightest )
	# print("mask", mask)
	# result = cv.bitwise_and( blobs, blobs, mask=mask )
	# cv.imshow("mask", mask)
	# cv.imshow("result", result)
	
	grayBlobs = cv.cvtColor( bgrBlobs, cv.COLOR_BGR2GRAY )
	thresh = 10
	maxVal = 255
	blockSize = 11
	c = 2  # "just a constant subtracted from the mean"
	ret, thresholdBlobs = cv.threshold(grayBlobs, thresh, maxVal, cv.THRESH_BINARY)
	# thresholdBlobs = cv.adaptiveThreshold(grayBlobs, maxVal, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
	# 									  cv.THRESH_BINARY, blockSize, c)
	cv.imshow("threshold", thresholdBlobs)
	im2, contours, hierarchy = cv.findContours( thresholdBlobs, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE )
	# for i in range(len(contours)):
	# 	cv.drawContours(frameCopy, contours, i, (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)))
	# cv.imshow("initial frame", frameCopy)
	
	contour = getContourFromInteriorPoint( click1, contours, hierarchy )
	cv.drawContours( frameCopy, [ contour ], 0, green )
	cv.imshow( "initial frame", frameCopy )
	
	cv.waitKey(0)
	
	# close window
	cv.destroyWindow( "initial frame" )
	
	return contour

def getBlobImage(frame1, frame2):
	hsvBlobs = np.zeros_like( frame1 )
	hsvBlobs[ ..., 1 ] = 255  # give all pixels 100% saturation
	
	prevGray = cv.cvtColor( frame1, cv.COLOR_BGR2GRAY )
	nextGray = cv.cvtColor( frame2, cv.COLOR_BGR2GRAY )
	flow = cv.calcOpticalFlowFarneback( prevGray, nextGray, None, 0.5, 3, 15, 3, 5, 1.2, 0 )
	
	# get flow vector info to turn into HSV color info
	mag, ang = cv.cartToPolar(flow[...,0], flow[...,1]) # output: vector magnitudes, angles

	# if flow is big enough
	# (ignores frames w/ almost no motion where very small flow values get normalized way too high,
	# resulting in frames w/ confetti)
	print("mag max", mag.max())
	if mag.max() > 5.:
		# vector angle => hue
		angDegrees = ang*180/np.pi/2 # convert radians to degrees
		hsvBlobs[...,0] = angDegrees

		# vector magnitude => value
		magNormalized = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
		magThreshold = 5.
		mask = magNormalized > magThreshold # at each index, boolean value for whether > threshold

		# uncomment for displayable version of mask
		# mask = mask.astype(np.float)
		# maskNorm = mask * 255
		# cv.imshow( 'blob mask', maskNorm )

		for i in range(mask.shape[0]):
			for j in range(mask.shape[1]):
				if mask[i][j]:
					hsvBlobs[i,j,2] = magNormalized[i,j]
	
	#bgrBlobs = cv.cvtColor( hsvBlobs, cv.COLOR_HSV2BGR )
	
	return hsvBlobs

def main( argv ):
	# check for command-line argument
	if len( argv ) < 2:
		print( "Usage: python3 main.py [video filename]" )
		exit( )
	
	cap = cv.VideoCapture( argv[ 1 ] )
	
	# check to make sure video opened successfully
	if not cap.isOpened( ):
		print( "Error opening video stream or file" )
		exit( -1 )
	
	# read in first frame
	ret, frame1 = cap.read( )
	#prevGray = cv.cvtColor( frame1, cv.COLOR_BGR2GRAY )
	
	# read in second frame
	ret, frame2 = cap.read( )
	# nextGray = cv.cvtColor( frame2, cv.COLOR_BGR2GRAY )
	
	# get flow between first 2 frames
	blobImage = getBlobImage(frame1, frame2)
	#cv.imshow("blobs", blobImage)
	#cv.waitKey(0)
	#input("wait")
	
	contourToTrack = selectObjects( frame1, blobImage ) #TODO: currently just 1 contour
	input( "AHA" )  # temp just for pausing
	prevGray = cv.cvtColor( frame2, cv.COLOR_BGR2GRAY ) # second frame => new start frame
	
	# image that all of the text gets accumulated in
	textImage = np.zeros_like(frame1)
	distanceTravelled = 0.0 # distance travelled since last text insertion
	
	# initialize array for HSV representation of flow blobs
	# hsvBlobs = np.zeros_like( frame1 )
	# hsvBlobs[ ..., 1 ] = 255  # give all pixels 100% saturation
	i = 0
	
	# loop through frames until video is completed
	while cap.isOpened( ):
		ret, frame2 = cap.read( )
		
		# stop immediately on a non-existent frame
		if frame2 is None:
			break
		
		nextGray = cv.cvtColor( frame2, cv.COLOR_BGR2GRAY )
		flow = cv.calcOpticalFlowFarneback( prevGray, nextGray, None, 0.5, 3, 15, 3, 5, 1.2, 0 )
		
		# find top 10 furthest points of contour
		# get their flow, average it to calc distance, & update all contour locations
		
		#TODO: aaaaaaaaaa
		
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
		
		# bgrBlobs = cv.cvtColor( hsvBlobs, cv.COLOR_HSV2BGR )
		# cv.imshow( 'BLOBS', bgrBlobs ) # uncomment to display just blobs
		# display = cv.add( frame2, bgrBlobs )  # draw flow blobs over original frame
		# cv.imshow( 'optical flow', display )
		cv.imshow("aaaa", frame2)
		
		# print("frame", i)
		# i += 1
		# k = cv.waitKey(1000) & 0xff
		k = cv.waitKey( 30 ) & 0xff
		# if k == 27:
		# 	break
		# elif k == ord('s'): # save frames on key press
		# 	cv.imwrite('opticalfb.png', frame2)
		# 	cv.imwrite('opticalhsv.png', bgr)
		
		prevGray = nextGray
	
	# clean up
	cap.release( )
	cv.destroyAllWindows( )


if __name__ == "__main__":
	main( sys.argv )