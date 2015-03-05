
#ifndef CAR_H
#define CAR_H


#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <numeric>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "gtest/gtest.h"

using namespace cv;

class Car
{
 public:
  Car(Point uL, Point lR, Mat initialIm);
  ~Car();
  
  void updateBoxPos(Mat image1, Mat image2);
  void updateBoxSize(Mat image);
  bool unitTest(Mat image);

 private:
  Point upperLeft;
  Point lowerRight;
  bool inertialTracking; 
  Mat templateIm;
  int updateTemplate;
 
  std::vector<Mat> confirmedFeatures;
  std::vector<int> gradientHistory;
  std::vector<int> xHistory;
  std::vector<int> yHistory;

  void checkBounds(Mat image, Point* uL, Point* lR);
  void getTemplateMatch(Mat image, Point* matchUL, Point* matchLR, Mat* templateImage);
  void matchPoints(std::vector<KeyPoint>* keypointsIn1, std::vector<KeyPoint>* keypointsIn2,Mat image1, Mat image2, std::vector<KeyPoint>* keypointsOut1, std::vector<KeyPoint>* keypointsOut2, int thresh);
  

};


#endif
