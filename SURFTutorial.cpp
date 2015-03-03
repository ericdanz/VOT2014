#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

void readme();

/** @function main */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_1.data || !img_2.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector.detect( img_1, keypoints_1 );
  detector.detect( img_2, keypoints_2 );

  //-- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;
 
  int minX = 5;
  int maxX = 53;

  int minY = 168;
  int maxY = 190;

  Point upperLeft(minX,minY);
  Point lowerRight(maxX,maxY);
   
  //std::cout<<keypoints_1.size() << std::endl;

  std::vector<KeyPoint> keypoints_car1;
  for (int i = 0; i < keypoints_1.size(); i++)
    {
      if (keypoints_1[i].pt.x < maxX && keypoints_1[i].pt.x > minX &&
	  keypoints_1[i].pt.y < maxY && keypoints_1[i].pt.y > minY)
	{
	  keypoints_car1.push_back(keypoints_1[i]);
	}

    }

  drawKeypoints( img_1, keypoints_car1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  rectangle(img_keypoints_1,upperLeft,lowerRight,10);
  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );
  imshow("Keypoints 2", img_keypoints_2 );

  waitKey(0);

  return 0;
  }

  /** @function readme */
  void readme()
  { std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl; }
