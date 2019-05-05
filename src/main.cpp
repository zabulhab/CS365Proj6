/* arSystem.cpp
 * Reads in camera calibration parameters and uses them to detect a chessboard
 * in video input and project objects onto the video input
 * 
 * to compile:
 * make main
 * 
 * Melody Mao & Zena Abulhab
 * CS365 Spring 2019
 * Project 4
 */

#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <vector>
//#include <algorithm>
#include <numeric>
#include <ctype.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int openVidFile(const char* vidName)
{

    cout << "Opening video file " << string(vidName) << "\n";
    
    VideoCapture *savedVid = new cv::VideoCapture(vidName);

    cout << "checking whether open\n";

    // open the video device
	if( !savedVid->isOpened() ) {
		printf("Unable to open video file %s\n", vidName);
		return(-1);
	}

	cv::Size refS( (int) savedVid->get(CAP_PROP_FRAME_WIDTH ),
		       (int) savedVid->get(CAP_PROP_FRAME_HEIGHT));

	printf("Expected size: %d %d\n", refS.width, refS.height);

    // Parameters for lucas-kanade method
    Size winSize  = Size(15,15);
    int maxLevel = 2;
    TermCriteria criteria = TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 0.03);


	namedWindow("Video", 1);

    // read the first frame to get the initial corner coordinates
    Mat oldGrayFrame;
    savedVid->read(oldGrayFrame);
    cvtColor(oldGrayFrame, oldGrayFrame, COLOR_BGR2GRAY);

	Mat frame, grayFrame;

    vector<Point2f> oldPoints;

    // feature params for good features to track
    int maxCorners = 100;
    double qualityLevel = 0.3;
    double minDistance = 7;
    int blockSize = 7;

    vector<Point2f> oldPointLocations;

    goodFeaturesToTrack(oldGrayFrame, oldPointLocations, maxCorners, qualityLevel, minDistance);

    //                    B  G  R
    Scalar green = Scalar(0,255,0); // for line color


	for(;;) {
		// get a new frame from the camera
        if (savedVid->read(frame) == false)
        {
            cout << "frame empty\n";
            break;            
        }

        
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        // Parameters for Lucas Kanade optical flow
        vector<Point2f> newPointLocations;
        vector<unsigned char> status;
        vector<float> error;



        calcOpticalFlowPyrLK(oldGrayFrame, grayFrame, oldPointLocations, newPointLocations, 
                                            status, error, winSize, maxLevel, criteria);


        // The points we actually found flow between
        vector<Point2f> goodNewPoints;
        vector<Point2f> goodOldPoints;



        for (int i = 0; i < oldPointLocations.size(); i++)
        {
            if (status[i] == 1)
            {
                goodNewPoints.push_back(newPointLocations[i]);
                goodOldPoints.push_back(oldPointLocations[i]);
            }
        }

        // draw the tracks
        for (int i = 0; i < goodOldPoints.size(); i++)
        {
            Point2f lineStart, lineEnd; //       thickness
            lineStart = goodOldPoints[i];
            lineEnd = goodNewPoints[i];
            line( frame, lineStart, lineEnd, green, 2 );
        }

        imshow("Video", frame);

        //check for user keyboard input
        char key = waitKey(10);
        
		if(key == 'q') 
        {
		    break;
		}

        // Make the current frame the old frame in the next iteration
        oldGrayFrame = Mat(grayFrame);
        // Update the coordinates of the old good points
        oldPointLocations = goodNewPoints;
	}

    delete savedVid;

    return (0);
}

int main(int argc, char *argv[])
{
    char vidName[256];
	// If user didn't give parameter file name
	if(argc < 1) 
	{
		cout << "Usage: ../bin/main |video file name|\n";
		exit(-1);
	}

    if (argc == 2) 
    {
        strcpy(vidName, argv[1]);

        // prerecorded video
        if (strstr(vidName, ".mp4") ||
            strstr(vidName, ".m4v") ||
            strstr(vidName, ".MOV") ||
            strstr(vidName, ".mov") ||
            strstr(vidName, ".avi") )
        {
            openVidFile(vidName);
        }
        else
        {
            cout << "Not a valid image or video extension\n";
        }
    }
    // else // live feed
    // {
    //     openVideoInput(cameraMatrix, distCoeffs);        
    // }

    return 0;
}