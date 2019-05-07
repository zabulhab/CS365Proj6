/* main.cpp
 * Reads in a video file and tracks optical flow
 * 
 * to compile:
 * make main
 * 
 * Melody Mao & Zena Abulhab
 * CS365 Spring 2019
 * Project 6
 */

#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include <numeric>
#include <ctype.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

struct CallbackParams
{
    vector<Point2f> *pointVectorOutput; //vector to store points into
    Mat *drawTo; //image to draw click points onto
};

Scalar green = Scalar(0,255,0); // for line/circle color

/**
 * Callback function for window created in getPointsToTrack,
 * which stores click locations into a specified vector and draws
 * dots there in a specified image, based on the given CallbackParams
 */
void mouseCallbackFunc(int event, int x, int y, int flags, void* userdata)
{
     CallbackParams *cbParams = (CallbackParams *)userdata;
     vector<Point2f> *clickPoints = cbParams->pointVectorOutput;
     Mat *frame = cbParams->drawTo;
     
     //on left click, store location
     if  ( event == EVENT_LBUTTONDOWN )
     {
          cout << "Storing point at (" << x << ", " << y << ")" << endl;
          clickPoints->push_back( Point2f(x, y) );

          //draw dot for click point
          circle(*frame, Point2f(x, y), 2, green, -1);
          imshow("Initial Frame", *frame);
     }
}

/**
 * Opens a window with the given video frame and prompts the user
 * to select points to track
 * Returns these selected points in the given output vector
 */
void getPointsToTrack(Mat &frame, vector<Point2f> &outputVector)
{
    Mat frameCopy(frame);

    cout << "Displaying initial frame; please select points to track by clicking\n";
    cout << "Press 'd' to complete point selection\n";

    namedWindow("Initial Frame", 1);

    CallbackParams cbParams;
    cbParams.pointVectorOutput = &outputVector;
    cbParams.drawTo = &frame;
    setMouseCallback("Initial Frame", mouseCallbackFunc, &cbParams); //set the callback function for any mouse event

    imshow("Initial Frame", frameCopy);

    while(true)
    {
        //check for user keyboard input
        char key = waitKey(10);
        if(key == 'd') 
        {
		    break;
		}
    }

    destroyWindow("Initial Frame");

    print(outputVector);
    cout << "\n";
}

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

	namedWindow("Video", 1);

    // read the first frame to get the initial corner coordinates
    Mat oldFrame, oldGrayFrame;
    savedVid->read(oldFrame);
    cvtColor(oldFrame, oldGrayFrame, COLOR_BGR2GRAY);

	Mat frame, grayFrame;

    // feature params for good features to track
    int maxCorners = 100;
    double qualityLevel = 0.3;
    double minDistance = 7;
    int blockSize = 7;

    vector<Point2f> oldPointLocations;

    getPointsToTrack(oldFrame, oldPointLocations);
    //goodFeaturesToTrack(oldGrayFrame, oldPointLocations, maxCorners, qualityLevel, minDistance);

    // Parameters for lucas-kanade method
    Size winSize = Size(15,15);
    int maxLevel = 2;
    TermCriteria criteria = TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 0.03);

    //                    B  G  R
    Scalar green = Scalar(0,255,0); // for line color

    //Mat to store pixel flow trails in, to draw over each frame
    Mat trailImage = Mat::zeros(oldFrame.size(), oldFrame.type());

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

        // draw the new flow tracks into the trail image
        for (int i = 0; i < goodOldPoints.size(); i++)
        {
            Point2f lineStart, lineEnd;
            lineStart = goodOldPoints[i];
            lineEnd = goodNewPoints[i];
            line( trailImage, lineStart, lineEnd, green, 2 ); //thickness = 2
        }

        //draw the trail image onto the current frame
        add(frame, trailImage, frame);

        imshow("Video", frame);

        //check for user keyboard input
        char key = waitKey(10);
        
		if(key == 'q') 
        {
		    break;
		}

        // Make the current frame the old frame in the next iteration
        oldGrayFrame = grayFrame.clone();
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
