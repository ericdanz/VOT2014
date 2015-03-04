#include "car.h"
using namespace cv;

int main( int argc, char** argv )
{
  //-- Initialize the first car location
  Point upperLeft(5,168);
  Point lowerRight(53,190);
   
  Mat initial_img = imread("../car/00000001.jpg", CV_LOAD_IMAGE_GRAYSCALE );
     
  //-- Initialize a car object 
  Car firstCar(upperLeft,lowerRight,initial_img);

  //-- Iterate through images 
  for (int j = 1; j < 252; j++)
    {
      char im1_name[30];
      char im2_name[30];
      sprintf(im1_name,"../car/00000%03d.jpg",j);
      sprintf(im2_name,"../car/00000%03d.jpg",j+1);
      Mat img_1 = imread( im1_name, CV_LOAD_IMAGE_GRAYSCALE );
      Mat img_2 = imread( im2_name, CV_LOAD_IMAGE_GRAYSCALE );
      printf("%s\n",im1_name);
      
      firstCar.updateBoxSize(img_1);

      firstCar.updateBoxPos(img_1, img_2);

    }
  return 0;
}

