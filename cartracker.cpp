//Written by Eric Danziger

#include "car.h"
using namespace cv;

int main( int argc, char** argv )
{
  //-- Initialize the first car location
  Point upperLeft(5,168);
  Point lowerRight(53,190);
   
  Mat initialImg = imread("../car/00000001.jpg", CV_LOAD_IMAGE_GRAYSCALE );
     
  //-- Initialize a car object 
  Car firstCar(upperLeft,lowerRight,initialImg);

  //-- Iterate through images 
  for (int j = 1; j < 252; j++)
    {
      char imName1[30];
      char imName2[30];
      sprintf(imName1,"../car/00000%03d.jpg",j);
      sprintf(imName2,"../car/00000%03d.jpg",j+1);
      Mat img1 = imread( imName1, CV_LOAD_IMAGE_GRAYSCALE );
      Mat img2 = imread( imName2, CV_LOAD_IMAGE_GRAYSCALE );
      printf("%s\n",imName1);
      
      firstCar.updateBoxSize(img1);

      firstCar.updateBoxPos(img1, img2);

    }
  return 0;
}

