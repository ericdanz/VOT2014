#include "car.h"
using namespace cv;

int main( int argc, char** argv )
{
  //-- Initialize the first car location
  int minX = 5;
  int maxX = 53;

  int minY = 168;
  int maxY = 190;

  Point upperLeft(minX,minY);
  Point lowerRight(maxX,maxY);
   
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
      fflush(stdout);
      
      firstCar.updateBoxSize(img_1);

      firstCar.updateBoxPos(img_1, img_2);

      //BEGIN // TEMPLATE BLOCK
      // Mat result;
      // int result_cols =  img_1.cols - firstCar.templateIm.cols + 1;
      // int result_rows = img_1.rows - firstCar.templateIm.rows + 1;

      // result.create( result_cols, result_rows, CV_32FC1 );

      // /// Do the Matching and Normalize
      // //matchTemplate( img_1, firstCar.templateIm, result, CV_TM_SQDIFF_NORMED );
      // matchTemplate( img_1, firstCar.templateIm, result, CV_TM_CCORR_NORMED );
      // normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

      // /// Localizing the best match with minMaxLoc
      // double minVal; double maxVal; Point minLoc; Point maxLoc;
      // Point matchLoc;

      // minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

      // /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better

      // //matchLoc = minLoc; 
      // matchLoc = maxLoc; 
  
      // /// Show me what you got
      // rectangle( img_1, matchLoc, Point( matchLoc.x + firstCar.templateIm.cols , matchLoc.y + firstCar.templateIm.rows ), Scalar::all(0), 2, 8, 0 );
      // imshow("templatebox",img_1);
      // waitKey(1);
      // //END TEMPLATE BLOCK

 
      usleep(1000000);

    }
  return 0;


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
 
 
  //std::cout<<keypoints_1.size() << std::endl;

  std::vector<KeyPoint> keypoints_car1, keypoints_gm1, keypoints_gm2;
  for (int i = 0; i < keypoints_1.size(); i++)
    {
      if (keypoints_1[i].pt.x < maxX && keypoints_1[i].pt.x > minX &&
	  keypoints_1[i].pt.y < maxY && keypoints_1[i].pt.y > minY)
	{
	  keypoints_car1.push_back(keypoints_1[i]);
	}

    }

  SurfDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor.compute(img_1, keypoints_car1, descriptors_1);
  extractor.compute(img_2, keypoints_2, descriptors_2);

  FlannBasedMatcher matcher;
  std::vector<DMatch> matches;
  matcher.match(descriptors_1,descriptors_2, matches);

  double max_dist = 0; double min_dist = 100;

   for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_1.rows; i++ )
  { if( matches[i].distance <= max(2*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

  for ( int i = 0; i < good_matches.size(); i++)
    {
      keypoints_gm2.push_back(keypoints_2[good_matches[i].trainIdx]);
      keypoints_gm1.push_back(keypoints_car1[good_matches[i].queryIdx]);
    }

  std::vector<double> xDistance, yDistance;
  

  for( int i = 0; i < (int)good_matches.size(); i++ )
  { 
    printf( "-- Good Match [%d] Keypoint 1x: %f  -- Keypoint 2x: %f  \n", i, keypoints_gm1[i].pt.x, keypoints_gm2[i].pt.x ); 
    xDistance.push_back(keypoints_gm2[i].pt.x - keypoints_gm1[i].pt.x);  
    yDistance.push_back(keypoints_gm2[i].pt.y - keypoints_gm1[i].pt.y); 
    printf( "-- Good Match [%d] xDistance: %f  -- yDistance: %f  \n", i, xDistance[i], yDistance[i] ); 
  }

  double avgX = std::accumulate(xDistance.begin(), xDistance.end(), 0) / (double)xDistance.size();
  double avgY = std::accumulate(yDistance.begin(), yDistance.end(), 0) / (double)yDistance.size();

  printf( "Distance x: %f  Distance y:  %f  \n", avgX, avgY );

  

  drawKeypoints( img_1, keypoints_gm1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( img_2, keypoints_gm2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  rectangle(img_keypoints_1,upperLeft,lowerRight,10);
  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );
  imshow("Keypoints 2", img_keypoints_2 );

  waitKey(0);

  return 0;
  }

