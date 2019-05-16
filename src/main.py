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
import math

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
def onMouseClick( event, x, y, flags, param ):
	frame = param[ "frame" ]
	boxPts = param[ "boxPts" ]
	
	if event == cv.EVENT_LBUTTONDOWN:  # drag start
		box = { "x1": int( x ), "y1": int( y ) }
		boxPts.append( box )
	elif event == cv.EVENT_LBUTTONUP:  # drag end
		thisBox = boxPts[ -1 ]
		thisBox[ "x2" ] = x
		thisBox[ "y2" ] = y
		
		#TODO: make it consistent for x1 < x2 and y1 < y2
		
		# draw box
		cv.line( frame, (thisBox[ "x1" ], thisBox[ "y1" ]),
				 (thisBox[ "x1" ], thisBox[ "y2" ]), green, thickness=2 )
		cv.line( frame, (thisBox[ "x1" ], thisBox[ "y1" ]),
				 (thisBox[ "x2" ], thisBox[ "y1" ]), green, thickness=2 )
		cv.line( frame, (thisBox[ "x1" ], thisBox[ "y2" ]),
				 (thisBox[ "x2" ], thisBox[ "y2" ]), green, thickness=2 )
		cv.line( frame, (thisBox[ "x2" ], thisBox[ "y1" ]),
				 (thisBox[ "x2" ], thisBox[ "y2" ]), green, thickness=2 )


# TODO: should this be for selecting multiple or just one?
'''
Handles initial phase when user drags a box on the first frame
to select the object to track
'''
def selectObjects( frame ):
	frameCopy = frame.copy( )
	
	cv.namedWindow( "initial frame" )
	boxPts = []
	param = { "boxPts": boxPts, "frame": frameCopy }
	cv.setMouseCallback( "initial frame", onMouseClick, param )
	
	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv.imshow( "initial frame", frameCopy )
		key = cv.waitKey( 1 ) & 0xFF
		
		# if the 'q' key is pressed, break from the loop
		if key == ord( "q" ):
			break
	
	# close window
	cv.destroyWindow( "initial frame" )
	
	return boxPts

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
	prevGray = cv.cvtColor( frame1, cv.COLOR_BGR2GRAY )
	
	# have user draw box to select object
	boxPts = selectObjects( frame1 )
	box = boxPts[0] # TODO: redo this when deciding whether to handle multiple boxes
	input( "AHA" )  # temp just for pausing
	
	# initialize array for HSV representation of flow blobs
	hsvBlobs = np.zeros_like( frame1 )
	hsvBlobs[ ..., 1 ] = 255  # give all pixels 100% saturation
	
	# image where we accumulate generated text to blit over each frame
	textImage = np.zeros_like( frame1 )
	
	distTraveledSinceLastText = 0.0
	
	#TODO: replace with actual image
	whiteness = np.full((50, 250, 3), 255, np.uint8)
	
	# loop through frames until video is completed
	while cap.isOpened( ):
		ret, frame2 = cap.read( )
		
		# stop immediately on a non-existent frame
		if frame2 is None:
			break
		
		#TODO: this is hardcoded for reaching the bottom
		# if object has reached bottom
		if box["y2"] >= frame2.shape[0]:
			break
				
		# calculate optical flow
		nextGray = cv.cvtColor( frame2, cv.COLOR_BGR2GRAY )
		flow = cv.calcOpticalFlowFarneback( prevGray, nextGray, None, 0.5, 3, 15, 3, 5, 1.2, 0 )
		
		# get flow vector info to turn into HSV color info
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1]) # output: vector magnitudes, angles

		# if flow is big enough
		# (ignores frames w/ almost no motion where very small flow values get normalized way too high,
		# resulting in frames w/ confetti)
		if mag.max() > 50.:
			# vector angle => hue
			angDegrees = ang*180/np.pi/2 # convert radians to degrees
			hsvBlobs[...,0] = angDegrees

			# vector magnitude => value
			hsvBlobs[...,2] = cv.normalize( mag, None, 0, 255, cv.NORM_MINMAX )
		#TODO: maybe an else for when the flow is really small
		
		# get ROI in blob image
		blobROI = hsvBlobs[box["y1"]:box["y2"], box["x1"]:box["x2"]]
		
		flattenedBlobROI = blobROI.reshape( (-1, 3) ) # reshape to array of 3-channel entries
		flattenedBlobROI = np.float32( flattenedBlobROI ) # convert to np.float32

		# define criteria, number of clusters(K) for k-means clustering
		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		K = 2
		ret, labels, clusterCenters = cv.kmeans( flattenedBlobROI, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS )

		# use k-means clustering for color segmentation of ROI
		clusterCenters = np.uint8( clusterCenters ) # convert cluster center colors back into uint8
		segmentedBlobROI = clusterCenters[ labels.flatten( ) ]
		segmentedBlobROI = segmentedBlobROI.reshape( blobROI.shape ) # reshape to region dimensions (un-flatten)
		
		# get most common color in blob image ROI
		(values, counts) = np.unique( labels, return_counts=True )
		idxMostCommonLabel = np.argmax( counts )
		idxMostCommonColor = values[ idxMostCommonLabel ]
		mostCommonColor = clusterCenters[idxMostCommonColor]
		
		#TODO: make sure most common color is not basically black (as in hand example)
		
		mask = cv.inRange( segmentedBlobROI, mostCommonColor, mostCommonColor )
		
		#TODO: this is just testing stuff
		# frameROI = frame2[box["y1"]:box["y2"], box["x1"]:box["x2"]]
		# result = cv.bitwise_and( frameROI, frameROI, mask=mask )
		# cv.imshow("is this the mask", result)
		# cv.waitKey(0)
		
		# get ROI in flow image
		flowROI = flow[box["y1"]:box["y2"], box["x1"]:box["x2"]]
		maskedFlow = flowROI[mask == 255]
		
		avgXFlow = np.mean( maskedFlow[...,0] )
		avgYFlow = np.mean( maskedFlow[...,1] )
		
		distTraveledSinceLastText += math.sqrt(avgXFlow * avgXFlow + avgYFlow * avgYFlow)
		print("traveled", distTraveledSinceLastText)
		
		# if another text instance will now fit
		if distTraveledSinceLastText > 50.0:
			# blit in text
			x1 = box["x1"]
			x2 = box["x1"] + whiteness.shape[1]
			y1 = box["y1"] - 50
			y2 = box["y1"]
			textImage[y1:y2, x1:x2] = whiteness
			distTraveledSinceLastText = 0.0
		
		# blit text image into frame and display it
		display = cv.add( frame2, textImage )
		cv.imshow( 'frame2', display )
		
		# move box by average flow
		box["x1"] = int(box["x1"] + avgXFlow)
		box["x2"] = int(box["x2"] + avgXFlow)
		box["y1"] = int(box["y1"] + avgYFlow)
		box["y2"] = int(box["y2"] + avgYFlow)
		
		# print("hallo", distTraveledSinceLastText)
		# input()
		
		'''
		- generate text at the end of the box, and find which side to anchor it relative to using lots of ifs and math mumbo-jumbo
		if points go out of bounds, try to handle that…somehow
			- figure out whether motion is more horizontal or vertical by checking if x or y flow is bigger in magnitude
			- figure out left vs right or up vs down by checking sign of that flow
				+x: moving right, -x: moving left
				+y: moving down, -y: moving up
			- text goes at opposite side. so if sign is positive, text goes at 0 (in x or y) + position of box
				if sign is negative, text goes at (width/height of box) + position of box
			- is text position defined as corner location or center location?
		
		where to resize text? b/c we only get flow in this loop
		but we only want to use it to determine box edge to align to and resize text on the first time
		'''
		
		# k = cv.waitKey(1000) & 0xff
		k = cv.waitKey( 30 ) & 0xff
		if k == ord('q'):
			break
		
		prevGray = nextGray
	
	# clean up
	cap.release( )
	cv.destroyAllWindows( )


if __name__ == "__main__":
	main( sys.argv )