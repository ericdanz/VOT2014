
#ifndef CAR_H
#define CAR_H


#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <numeric>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
using namespace cv;

class Car
{
 public:
Car(Point uL, Point lR);
  ~Car();
  
  Point upperLeft;
  Point lowerRight;
 
  std::vector<Mat> confirmedFeatures;
  std::vector<int> gradientHistory;
  
  void updateBoxPos(Mat image_1, Mat image_2);
  void updateBoxSize(Mat image);
  void checkBounds(Mat image);
 private:


};


#endif
