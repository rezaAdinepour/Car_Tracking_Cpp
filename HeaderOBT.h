#pragma once 
#include <opencv.hpp>

#ifndef HEADER
#define HEADER

#if _DEBUG
    #pragma comment(lib, "opencv_world460d.lib")
#else
    #pragma comment(lib, "opencv_world460.lib")
#endif

using namespace std;
using namespace cv;



class CarTracking
{
public:
    // member variables ///////////////////////////////////////////////////////////////////////////

    vector<Point> currentContour;

    Rect currentBoundingRect;

    vector<Point> centerPositions;

    double CurrentDiagonalSize;
    double CurrentAspectRatio;

    bool CurrentMatchFoundOrNewCar;

    bool StillBeingTracked;

    int nConsecutiveFramesWithoutAMatch;

    Point predictedNextPosition;

    // function prototypes ////////////////////////////////////////////////////////////////////////
    CarTracking(vector<Point> _contour);
    void predictNextPosition();

};

#endif    

