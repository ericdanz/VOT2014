
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
using namespace cv;

class Car
{
 public:
  Car(Point uL, Point lR, Mat initialIm);
  ~Car();
  
  Point upperLeft;
  Point lowerRight;
  bool inertialTracking; 
  Mat templateIm;
  int carConfidence;
 
  std::vector<Mat> confirmedFeatures;
  std::vector<int> gradientHistory;
  std::vector<int> xHistory;
  std::vector<int> yHistory;
  
  void updateBoxPos(Mat image_1, Mat image_2);
  void updateBoxSize(Mat image);
  void checkBounds(Mat image, Point* uL, Point* lR);

  void matchPoints(std::vector<KeyPoint>* kp_in1, std::vector<KeyPoint>* kp_in2,Mat image_1, Mat image_2, std::vector<KeyPoint>* kp_out1, std::vector<KeyPoint>* kp_out2, int thresh);

private:


};


#endif
