#include "car.h"
using namespace cv;

Car::Car(Point uL, Point lR)
{
  upperLeft = uL;
  lowerRight = lR;
}

Car::~Car()
{
}

void Car::updateBoxPos(Mat image_1, Mat image_2)
{

}

void Car::updateBoxSize(Mat image)
{


}

void Car::checkBounds(Mat image)
{
  //Fix the box points if they are outside of bounds
  if (upperLeft.x < 1) upperLeft.x = 1;
  if (upperLeft.y < 1) upperLeft.y = 1;
  if (lowerRight.x > image.size().width) lowerRight.x = image.size().width;
  if (lowerRight.y > image.size().height) lowerRight.y = image.size().height;
}
