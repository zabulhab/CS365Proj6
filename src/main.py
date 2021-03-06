"""
main.py
Opens a video file, tracks object motion in it,
and generates text in the wake of the moving object(s)
Adapted from the example: https://docs.opencv.org/3.4/d7/d8b/tutorial_py_lucas_kanade.html

to run:
python3 main.py [video filename] [text filename] [-w]
The -w is optional. When present, it indicates that the user would like to write
the text generation result out to a video file.

Zena Abulhab & Melody Mao
CS365 Spring 2019
Project 6
"""

import sys
import cv2 as cv
import numpy as np
import math

green = (0, 255, 0)
black = (0, 0, 0)

'''
Mouse click callback function for initial frame window
Detects a box dragged by the user and draws it onto the frame
'''
def onMouseClick( event, x, y, flags, param ):
	frame = param[ "frame" ]
	thisBox = param[ "boxPts" ]
	
	if event == cv.EVENT_LBUTTONDOWN:  # drag start
		thisBox[ "col1" ] = x
		thisBox[ "row1" ] = y
	elif event == cv.EVENT_LBUTTONUP:  # drag end
		thisBox[ "col2" ] = x
		thisBox[ "row2" ] = y
		
		# ensure col1 < col2 and row1 < row2; swap if not
		if thisBox["col1"] > thisBox["col2"]:
			thisBox["col2"] = thisBox["col1"]
			thisBox["col1"] = x
		if thisBox["row1"] > thisBox["row2"]:
			thisBox["row2"] = thisBox["row1"]
			thisBox["row1"] = y
				
		# draw box
		cv.line( frame, (thisBox[ "col1" ], thisBox[ "row1" ]),
				 (thisBox[ "col1" ], thisBox[ "row2" ]), green, thickness=2 )
		cv.line( frame, (thisBox[ "col1" ], thisBox[ "row1" ]),
				 (thisBox[ "col2" ], thisBox[ "row1" ]), green, thickness=2 )
		cv.line( frame, (thisBox[ "col1" ], thisBox[ "row2" ]),
				 (thisBox[ "col2" ], thisBox[ "row2" ]), green, thickness=2 )
		cv.line( frame, (thisBox[ "col2" ], thisBox[ "row1" ]),
				 (thisBox[ "col2" ], thisBox[ "row2" ]), green, thickness=2 )

'''
Handles initial phase when user drags a box on the first frame
to select the object to track
'''
def selectObjects( frame ):
	frameCopy = frame.copy( ) # copy of frame to draw boxes onto
	
	cv.namedWindow( "initial frame" )
	boxPts = {}
	param = { "boxPts": boxPts, "frame": frameCopy }
	cv.setMouseCallback( "initial frame", onMouseClick, param )
	
	# keep looping until the 'q' key is pressed
	while True:
		# display the image
		cv.imshow( "initial frame", frameCopy )
		
		# if the 'q' key is pressed, break from the loop
		key = cv.waitKey( 1 ) & 0xFF
		if key == ord( "q" ):
			break
	
	# close window
	cv.destroyWindow( "initial frame" )
	
	return boxPts

'''
Calculates the average of the most common optical flow range
between the given gray-scale frames within the given box,
using the given (HSV) flow blob image
'''
def calcAvgFlow(prevGray, nextGray, hsvBlobs, box):
	flow = cv.calcOpticalFlowFarneback( prevGray, nextGray, None, 0.5, 3, 15, 3, 5, 1.2, 0 )
	
	# get flow vector info to turn into HSV color info
	mag, ang = cv.cartToPolar( flow[ ..., 0 ], flow[ ..., 1 ] )  # output: vector magnitudes, angles
	
	# if flow is big enough
	# (ignores frames w/ almost no motion where very small flow values get normalized way too high,
	# resulting in frames w/ confetti)
	if mag.max( ) > 50.:
		# vector angle => hue
		angDegrees = ang * 180 / np.pi / 2  # convert radians to degrees
		hsvBlobs[ ..., 0 ] = angDegrees
		
		# vector magnitude => value
		hsvBlobs[ ..., 2 ] = cv.normalize( mag, None, 0, 255, cv.NORM_MINMAX )
	else:
		hsvBlobs[...,2] = 0
	
	# get ROI in blob image
	blobROI = hsvBlobs[ box[ "row1" ]:box[ "row2" ], box[ "col1" ]:box[ "col2" ] ]
	
	flattenedBlobROI = blobROI.reshape( (-1, 3) )  # reshape to array of 3-channel entries
	flattenedBlobROI = np.float32( flattenedBlobROI )  # convert to np.float32
	
	# define criteria, number of clusters(K) for k-means clustering
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = 2
	ret, labels, clusterCenters = cv.kmeans( flattenedBlobROI, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS )
	
	# use k-means clustering for color segmentation of ROI
	clusterCenters = np.uint8( clusterCenters )  # convert cluster center colors back into uint8
	segmentedBlobROI = clusterCenters[ labels.flatten( ) ]
	segmentedBlobROI = segmentedBlobROI.reshape( blobROI.shape )  # reshape to region dimensions (un-flatten)
	
	# get most common color in blob image ROI
	(values, counts) = np.unique( labels, return_counts=True )
	idxMostCommonLabel = np.argmax( counts )
	idxMostCommonColor = values[ idxMostCommonLabel ]
	mostCommonColor = clusterCenters[ idxMostCommonColor ]
	
	mask = cv.inRange( segmentedBlobROI, mostCommonColor, mostCommonColor )
	
	# get ROI in flow image
	flowROI = flow[ box[ "row1" ]:box[ "row2" ], box[ "col1" ]:box[ "col2" ] ]
	maskedFlow = flowROI[ mask == 255 ]
	
	# calculate averages
	avgXFlow = np.mean( maskedFlow[ ..., 0 ] )
	avgYFlow = np.mean( maskedFlow[ ..., 1 ] )
	
	return avgXFlow, avgYFlow

'''
Rotates the given image by the given angle without cutting the rotated version off
to the original dimensions
code from: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
'''
def rotate_bound(image, angle):
	# grab the dimensions of the image and then determine the
	# center
	(h, w) = image.shape[:2]
	(cX, cY) = (w // 2, h // 2)

	# grab the rotation matrix (applying the negative of the
	# angle to rotate clockwise), then grab the sine and cosine
	# (i.e., the rotation components of the matrix)
	M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])

	# compute the new bounding dimensions of the image
	nW = int((h * sin) + (w * cos))
	nH = int((h * cos) + (w * sin))

	# adjust the rotation matrix to take into account translation
	M[0, 2] += (nW / 2) - cX
	M[1, 2] += (nH / 2) - cY

	# perform the actual rotation and return the image
	return cv.warpAffine(image, M, (nW, nH))

'''
Determines the generated text's orientation and position (relative to the tracking box)
from the given average initial flow, tracking box, and text image
Returns the distance to travel before blitting in text,
the position of the text image (its upper left corner) relative to the tracking box,
the updated text image, and the dimensions of the box the text image is drawn in
'''
def getTextDirAndPos( avgXFlow, avgYFlow, box, text ):
	xFlowMag = abs( avgXFlow )
	yFlowMag = abs( avgYFlow )
	
	boxWidth = box[ "col2" ] - box[ "col1" ]
	boxHeight = box[ "row2" ] - box[ "row1" ]
	
	# figure out whether motion is more horizontal or vertical by checking if x or y flow is bigger in magnitude
	if xFlowMag > yFlowMag:  # horizontal
		textRowPosFromBox = 0  # top of text lines up with top of box
		
		# resize text image width to box height
		textWidth = boxHeight
		textHeight = int( text.shape[ 0 ] * textWidth / text.shape[ 1 ] )
		text = cv.resize( text, (textWidth, textHeight) )
		
		# swap width and height because of rotation
		temp = textWidth
		textWidth = textHeight
		textHeight = temp
		
		distToTravel = textWidth
		
		if avgXFlow < 0.0:  # moving left
			textColPosFromBox = boxWidth  # left edge of text is at right edge of box
			
			# rotate text 90 degrees counterclockwise
			text = rotate_bound( text, -90 )
		else:  # moving right
			textColPosFromBox = - textWidth  # left edge of text is 1 text-width away from left edge of box
			
			# rotate text 90 degrees clockwise
			text = rotate_bound( text, 90 )
	
	else:  # vertical
		textColPosFromBox = 0
		
		# resize text image width to box width
		textWidth = boxWidth
		textHeight = int( text.shape[ 0 ] * textWidth / text.shape[ 1 ] )
		text = cv.resize( text, (textWidth, textHeight) )
		
		distToTravel = textHeight
		
		# note: no need to rotate text
		
		if avgYFlow < 0.0:  # moving up
			textRowPosFromBox = boxHeight # top of text lines up with bottom of box
		else:  # moving down
			textRowPosFromBox = - textHeight # top of text is 1 text-height above top of box
	
	# ensure both relative positions are integer values
	textRowPosFromBox = int( textRowPosFromBox )
	textColPosFromBox = int( textColPosFromBox )
	return distToTravel, textColPosFromBox, textRowPosFromBox, text, textWidth, textHeight

def main( argv ):
	# check for command-line argument
	if len( argv ) < 3:
		print( "Usage: python3 main.py [video filename] [text filename] [-w]" )
		exit( )
	
	# read in text image
	text = cv.imread(argv[2])
	if text is None:
		print("Could not read in image:", argv[2])
		exit()
	
	# open video file
	cap = cv.VideoCapture( argv[1] )
	
	# check to make sure video opened successfully
	if not cap.isOpened( ):
		print( "Error opening video stream or file" )
		exit( -1 )
	
	# check whether to write out video
	writeVideo = False
	if len(argv) > 3 and argv[3] == "-w":
		writeVideo = True
	
	# read in first frame
	ret, frame1 = cap.read( )
	prevGray = cv.cvtColor( frame1, cv.COLOR_BGR2GRAY )
	
	# have user draw box to select object
	box = selectObjects( frame1 )
	input( "Press enter to start text generation" )
	
	# initialize array for HSV representation of flow blobs
	hsvBlobs = np.zeros_like( frame1 )
	hsvBlobs[ ..., 1 ] = 255  # give all pixels 100% saturation
	
	# image where we accumulate generated text to blit over each frame
	textImage = np.zeros_like( frame1 )
	
	distTraveledSinceLastText = 0.0
	
	# read in second frame to figure out which side the text should go on and resize text accordingly
	ret, frame2 = cap.read( )
	# stop immediately on a non-existent frame
	if frame2 is None:
		return

	# calculate optical flow
	nextGray = cv.cvtColor( frame2, cv.COLOR_BGR2GRAY )
	avgXFlow, avgYFlow = calcAvgFlow( prevGray, nextGray, hsvBlobs, box )

	distTraveledSinceLastText += math.sqrt( avgXFlow * avgXFlow + avgYFlow * avgYFlow )

	cv.imshow( 'frame2', frame2 )
	
	# move box by average flow
	box[ "col1" ] = int( box[ "col1" ] + avgXFlow )
	box[ "col2" ] = int( box[ "col2" ] + avgXFlow )
	box[ "row1" ] = int( box[ "row1" ] + avgYFlow )
	box[ "row2" ] = int( box[ "row2" ] + avgYFlow )
	
	distToTravel, textColPosFromBox, textRowPosFromBox, text, textWidth, textHeight = getTextDirAndPos( avgXFlow,
																										avgYFlow, box,
																										text )
	
	# initialize video writer
	videoOut = None
	if writeVideo:
		frame_width = int( cap.get( 3 ) )
		frame_height = int( cap.get( 4 ) )
		videoOut = cv.VideoWriter( 'result.avi', cv.VideoWriter_fourcc( 'M', 'J', 'P', 'G' ),
								   10, (frame_width,frame_height) )
		# Write the frame into the file
		videoOut.write( frame1 )
		videoOut.write( frame2 )
	
	# loop through frames until video is completed
	while cap.isOpened( ):
		ret, frame2 = cap.read( )
		
		# stop immediately on a non-existent frame
		if frame2 is None:
			break
		
		# check if object is about to go out of frame
		if box["row1"] <= 0: # if object has reached top
			break
		if box["row2"] >= frame2.shape[0]: # if object has reached bottom
			break
		if box["col1"] <= 0: # if object has reached left edge
			break
		if box["col2"] >= frame2.shape[1]: # if object has reached right edge
			break
				
		# calculate optical flow
		nextGray = cv.cvtColor( frame2, cv.COLOR_BGR2GRAY )
		avgXFlow, avgYFlow = calcAvgFlow(prevGray, nextGray, hsvBlobs, box)
		
		distTraveledSinceLastText += math.sqrt(avgXFlow * avgXFlow + avgYFlow * avgYFlow)
		
		# if another text instance will now fit
		if distTraveledSinceLastText > distToTravel * 2:
			# blit in text
			col1 = box["col1"] + textColPosFromBox
			col2 = box["col1"] + textColPosFromBox + textWidth
			row1 = box["row1"] + textRowPosFromBox
			row2 = box["row1"] + textRowPosFromBox + textHeight
			
			textImage[row1:row2, col1:col2] = text
			
			distTraveledSinceLastText = 0.0
		
		# blit text image into frame and display it
		textMask = 255 - cv.inRange( textImage, black, black )
		textMask = np.where(textMask != 0)
		frame2[ textMask[0], textMask[1] ] = textImage[ textMask[0], textMask[1]]
		
		cv.imshow( 'frame2', frame2 )
		
		# write frame into video file
		if writeVideo:
			videoOut.write(frame2)
		
		# move box by average flow
		box["col1"] = int(box["col1"] + avgXFlow)
		box["col2"] = int(box["col2"] + avgXFlow)
		box["row1"] = int(box["row1"] + avgYFlow)
		box["row2"] = int(box["row2"] + avgYFlow)
		
		k = cv.waitKey( 30 ) & 0xff
		if k == ord('q'):
			break
		
		prevGray = nextGray
	
	# clean up
	cap.release( )
	videoOut.release()
	cv.destroyAllWindows( )


if __name__ == "__main__":
	main( sys.argv )