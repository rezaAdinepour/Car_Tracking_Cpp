// Object Tracking.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv.hpp>
#include "HeaderOBT.h"

#if _DEBUG
#pragma comment(lib, "opencv_world460d.lib")
#else
#pragma comment(lib, "opencv_world460.lib")
#endif

using namespace std;
using namespace cv;

#define SHOW_STEPS


// global variables ///////////////////////////////////////////////////////////////////////////////

const Scalar BLUE = Scalar(255, 0, 0);
const Scalar BLACK = Scalar(0, 0, 0);
const Scalar WHITE = Scalar(255, 255, 255);
const Scalar YELLOW = Scalar(0, 255, 255);
const Scalar GREEN = Scalar(0, 200, 0);
const Scalar RED = Scalar(0, 0, 255);

// function prototypes ////////////////////////////////////////////////////////////////////////////

void matchCurrentFrameToExistingCar(vector<CarTracking>& existingCars, vector<CarTracking>& currentFrameCars);
void addCarToExistingCars(CarTracking& currentFrameCar, vector<CarTracking>& existingCars, int& intIndex);
void addNewCar(CarTracking& currentFrameCar, vector<CarTracking>& existingCars);
double distanceBetweenPoints(Point point1, Point point2);
void drawAndShowContours(Size imageSize, vector<vector<Point> > contours, string strImageName);
void drawAndShowContours(Size imageSize, vector<CarTracking> Cars, string strImageName);
bool checkIfCarsCrossedTheLine(vector<CarTracking>& Cars, int& intHorizontalLinePosition, int& CarCountIn, int& CarCountOut);
void drawCarInfoOnImage(vector<CarTracking>& Cars, Mat& imgFrame2Copy);
void drawCarCountOnImage(int& CarCountIn, int& CarCountOut, Mat& imgFrame2Copy);

// main starts here///////////////////////////////////////////////////////////////////////////////////////////////////

int main() 
{

    VideoCapture cap;

    Mat imgFrame1;
    Mat imgFrame2;

    vector <CarTracking> Cars;

    int CarCountIn = 0;
    int CarCountOut = 0;

    // open video file
    cap.open("Relaxing_highway_traffic.mp4", 0);
    //cap.open("CarsDrivingUnderBridge.mp4", 0);

    // if video can not open
    if (!cap.isOpened())
    {                                                       
        cout << "Error! the Video file cannot open" << endl;
        return(0);                                          
    }

    //recive frame
    cap >> imgFrame1;
    cap >> imgFrame2;

    //line counting
    int HorizontalLinePosition = (int) round((double)imgFrame1.rows * 0.7);

    Point crossingLine[2];
    crossingLine[0].x = 0; //x pixel from left side
    crossingLine[0].y = HorizontalLinePosition; //y pixel from left side

    crossingLine[1].x = imgFrame1.cols - 1;
    crossingLine[1].y = HorizontalLinePosition;

    char CheckForEscKey = 0;                            // for exit of program

    bool FirstFrame = true;

    int frameCount = 2;

    while (cap.isOpened() && CheckForEscKey != 27)
    {

        vector<CarTracking> currentFrame;

        Mat imgFrame1Copy = imgFrame1.clone();
        Mat imgFrame2Copy = imgFrame2.clone();
        Mat imgDiff, imgThresh;

        cvtColor(imgFrame1Copy, imgFrame1Copy, COLOR_BGR2GRAY);
        cvtColor(imgFrame2Copy, imgFrame2Copy, COLOR_BGR2GRAY);

        GaussianBlur(imgFrame1Copy, imgFrame1Copy, Size(5, 5), 0);
        GaussianBlur(imgFrame2Copy, imgFrame2Copy, Size(5, 5), 0);

        absdiff(imgFrame1Copy, imgFrame2Copy, imgDiff);
        //imshow("imgBlur", imgDiff);


        threshold(imgDiff, imgThresh, 30, 255, THRESH_BINARY);
        //imshow("imgThresh", imgThresh);


        Mat structuringElement3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));

        for (int i = 0; i < 2; i++)
        {
            dilate(imgThresh, imgThresh, structuringElement3x3);
            dilate(imgThresh, imgThresh, structuringElement3x3);
            erode(imgThresh, imgThresh, structuringElement3x3);

        }

        Mat imgThreshCopy = imgThresh.clone();

        vector<vector<Point>> contours;

        findContours(imgThreshCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        drawAndShowContours(imgThresh.size(), contours, "imgContours");

        vector<vector <Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++)
        {
            convexHull(contours[i], convexHulls[i]);
        }

        drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

        for (auto& convexHull : convexHulls)
        {
            CarTracking possibleCar(convexHull);

            if (possibleCar.currentBoundingRect.area() > 400 &&
                possibleCar.CurrentAspectRatio > 0.2 &&
                possibleCar.CurrentAspectRatio < 4 &&
                possibleCar.currentBoundingRect.width > 30 &&
                possibleCar.currentBoundingRect.height > 30 &&
                possibleCar.CurrentDiagonalSize > 60 &&
                (contourArea(possibleCar.currentContour) / (double)possibleCar.currentBoundingRect.area()) > 0.5) {
                currentFrame.push_back(possibleCar);
            }
        }

        drawAndShowContours(imgThresh.size(), currentFrame, "imgCurrentFrameCars");


        if (FirstFrame == true)
        {
            for (auto& currentFrameCar : currentFrame)
            {
                Cars.push_back(currentFrameCar);
            }
        }
        else
        {
            matchCurrentFrameToExistingCar(Cars, currentFrame);
        }

        drawAndShowContours(imgThresh.size(), Cars, "imgCars");

        imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

        drawCarInfoOnImage(Cars, imgFrame2Copy);

        bool nAtLeastOneCrossedTheLine = checkIfCarsCrossedTheLine(Cars, HorizontalLinePosition, CarCountIn, CarCountOut);

        if (nAtLeastOneCrossedTheLine == true)
        {
            line(imgFrame2Copy, crossingLine[0], crossingLine[1], GREEN, 2);
        }
        else
        {
            line(imgFrame2Copy, crossingLine[0], crossingLine[1], RED, 2);
        }

        drawCarCountOnImage(CarCountIn, CarCountOut, imgFrame2Copy);

        imshow("imgFrame2Copy", imgFrame2Copy);

        //waitKey(1);                

        // now we are prepare for the next iteration

        currentFrame.clear();

        imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

        if ((cap.get(CAP_PROP_POS_FRAMES) + 1) < cap.get(CAP_PROP_FRAME_COUNT))
        {
            cap >> imgFrame2;
        }
        else
        {
            cout << "End of video" << endl;
            break;
        }

        FirstFrame = false;
        frameCount++;
        CheckForEscKey = waitKey(1);
    }

    if (CheckForEscKey != 27)        // if the user did not press esc or we reached the end of the video
    {
        waitKey(0);                  // hold the windows open to allow the "end of video" message to show
    }


    return(0);
}

// function definitions ///////////////////////////////////////////////////////////////////////////////////////////////////

void matchCurrentFrameToExistingCar(vector<CarTracking>& existingCars, vector<CarTracking>& currentFrameCars) {

    for (auto& existingCar : existingCars)
    {

        existingCar.CurrentMatchFoundOrNewCar = false;

        existingCar.predictNextPosition();
    }

    for (auto& currentFrameCar : currentFrameCars)
    {

        int IndexofLeastDistance = 0;
        double LeastDistance = 100000;

        for (unsigned int i = 0; i < existingCars.size(); i++)
        {

            if (existingCars[i].StillBeingTracked == true)
            {
                double Distance = distanceBetweenPoints(currentFrameCar.centerPositions.back(), existingCars[i].predictedNextPosition);

                if (Distance < LeastDistance) {
                    LeastDistance = Distance;
                    IndexofLeastDistance = i;
                }
            }
        }

        if (LeastDistance < currentFrameCar.CurrentDiagonalSize * 0.5)
        {
            addCarToExistingCars(currentFrameCar, existingCars, IndexofLeastDistance);
        }
        else
        {
            addNewCar(currentFrameCar, existingCars);
        }

    }

    for (auto& existingCar : existingCars)
    {

        if (existingCar.CurrentMatchFoundOrNewCar == false)
        {
            existingCar.nConsecutiveFramesWithoutAMatch++;
        }

        if (existingCar.nConsecutiveFramesWithoutAMatch >= 5)
        {
            existingCar.StillBeingTracked = false;
        }

    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void addCarToExistingCars(CarTracking& currentFrameCar, vector<CarTracking>& existingCars, int& Index) {

    existingCars[Index].currentContour = currentFrameCar.currentContour;
    existingCars[Index].currentBoundingRect = currentFrameCar.currentBoundingRect;

    existingCars[Index].centerPositions.push_back(currentFrameCar.centerPositions.back());

    existingCars[Index].CurrentDiagonalSize = currentFrameCar.CurrentDiagonalSize;
    existingCars[Index].CurrentAspectRatio = currentFrameCar.CurrentAspectRatio;

    existingCars[Index].StillBeingTracked = true;
    existingCars[Index].CurrentMatchFoundOrNewCar = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void addNewCar(CarTracking& currentFrameCar, vector<CarTracking>& existingCars) {

    currentFrameCar.CurrentMatchFoundOrNewCar = true;

    existingCars.push_back(currentFrameCar);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

double distanceBetweenPoints(Point pt1, Point pt2) {

    int X = abs(pt1.x - pt2.x);
    int Y = abs(pt1.y - pt2.y);

    return(sqrt(pow(X, 2) + pow(Y, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void drawAndShowContours(Size imageSize, vector<vector<Point>> contours, string strImageName) {
    Mat image(imageSize, CV_8UC3, BLACK);

    drawContours(image, contours, -1, WHITE, -1);

    //imshow(strImageName, image);
    //waitKey(20);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void drawAndShowContours(Size imageSize, vector<CarTracking> Cars, string strImageName) {

    Mat image(imageSize, CV_8UC3, BLACK);

    vector<vector<Point>> contours;

    for (auto& Car : Cars)
    {
        if (Car.StillBeingTracked == true) {
            contours.push_back(Car.currentContour);
        }
    }

    drawContours(image, contours, -1, WHITE, -1);

    imshow(strImageName, image);
    //waitKey(20);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

bool checkIfCarsCrossedTheLine(vector<CarTracking>& Cars, int& HorizontalLinePosition, int& CarCountIn, int& CarCountOut) {
    bool AtLeastOneCarCrossedTheLine = false;

    for (auto Car : Cars)
    {

        if (Car.StillBeingTracked == true && Car.centerPositions.size() >= 2)
        {
            int prevFrameIndex = (int)Car.centerPositions.size() - 2;
            int currFrameIndex = (int)Car.centerPositions.size() - 1;

            if (Car.centerPositions[prevFrameIndex].y > HorizontalLinePosition &&
                Car.centerPositions[currFrameIndex].y <= HorizontalLinePosition)
            {
                CarCountIn++;
                AtLeastOneCarCrossedTheLine = true;
            }

            if (Car.centerPositions[prevFrameIndex].y < HorizontalLinePosition &&
                Car.centerPositions[currFrameIndex].y >= HorizontalLinePosition)
            {
                CarCountOut++;
                AtLeastOneCarCrossedTheLine = true;
            }
        }

    }

    return AtLeastOneCarCrossedTheLine;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void drawCarInfoOnImage(vector<CarTracking>& Cars, Mat& imgFrame2Copy) {

    for (unsigned int i = 0; i < Cars.size(); i++)
    {

        if (Cars[i].StillBeingTracked == true)
        {
            rectangle(imgFrame2Copy, Cars[i].currentBoundingRect, BLUE, 2);

            int FontFace = FONT_HERSHEY_COMPLEX_SMALL;
            double FontScale = Cars[i].CurrentDiagonalSize / 60;
            int FontThickness = (int)round(FontScale * 1);
            /*String str = "Car ";
            String str2 = to_string(i);
            str += str2;*/
            putText(imgFrame2Copy, to_string(i), Cars[i].centerPositions.back(),
                FontFace, FontScale, GREEN, FontThickness);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void drawCarCountOnImage(int& CarCountIn, int& CarCountOut, Mat& imgFrame2Copy) {

    int FontFace = FONT_HERSHEY_DUPLEX;
    double FontScale = 1.5;
    int FontThickness = (int)round(FontScale * 2);

    Size textRightSize = getTextSize(to_string(CarCountIn),
        FontFace, FontScale, FontThickness, 0);

    Size textLeftSize = getTextSize(to_string(CarCountOut),
        FontFace, FontScale, FontThickness, 0);

    Point TextBottomRightPosition;
    Point TextBottomLeftPosition;

    TextBottomRightPosition.x = imgFrame2Copy.cols - 150 -
        (int)((double)textRightSize.width * 1.25);
    TextBottomRightPosition.y = (int)((double)textRightSize.height * 1.25);

    TextBottomLeftPosition.x = 30;
    TextBottomLeftPosition.y = (int)((double)textRightSize.height * 1.25);

    String in = "Out: ";
    String out = "In: ";
    String str1 = to_string(CarCountIn);
    String str2 = to_string(CarCountOut);
    in += str1;
    out += str2;

    putText(imgFrame2Copy, in, TextBottomRightPosition,
        FontFace, FontScale, BLACK, FontThickness);
    putText(imgFrame2Copy, out, TextBottomLeftPosition,
        FontFace, FontScale, BLACK, FontThickness);
}