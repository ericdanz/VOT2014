//This is the test file
//This calls unitTest, which contains a basic set of unit tests
//
//Written by Eric Danziger
//04 March 2015
//ericdanziger@cmu.edu

#include "car.h"
using namespace cv;


int main( int argc, char** argv )
{
  Point upperLeft(5,168);
Point lowerRight(53,190);
Mat initialImg = imread("../car/00000001.jpg", CV_LOAD_IMAGE_GRAYSCALE );

Car testCar(upperLeft,lowerRight,initialImg);

//-- Run the unit test
bool result = testCar.unitTest(initialImg);

if(result) printf("\n[ PASS ]  ALL TESTS PASSED \n\n");
else printf("[ FAIL ]  ONE OR MORE TESTS FAILED \n\n"); 


 return 0;
}
