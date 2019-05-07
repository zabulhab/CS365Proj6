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
		
		#TODO: make trails
		# turn flow vector info into HSV color info
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
		hsvBlobs[...,0] = ang*180/np.pi/2 # angle => hue
		hsvBlobs[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX) # magnitude => value
		
		bgr = cv.cvtColor(hsvBlobs, cv.COLOR_HSV2BGR)
		display = cv.add(frame2, bgr) # draw flow blobs over original frame
		cv.imshow('frame2', display)
		
		k = cv.waitKey(30) & 0xff
		if k == 27:
			break
		elif k == ord('s'): # save frames on key press
			cv.imwrite('opticalfb.png', frame2)
			cv.imwrite('opticalhsv.png', bgr)
		
		prevGray = nextGray
	
	# clean up
	cap.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	main(sys.argv)