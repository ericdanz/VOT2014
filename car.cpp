#include "car.h"
using namespace cv;

Car::Car(Point uL, Point lR, Mat initialIm)
{

  upperLeft = uL;
  lowerRight = lR;
  inertialTracking = false;
  checkBounds(initialIm, &upperLeft, &lowerRight);
  templateIm = initialIm(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));
  updateTemplate = 1;  
}

Car::~Car()
{
}

void Car::matchPoints(std::vector<KeyPoint>* keypointsIn1, std::vector<KeyPoint>* keypointsIn2,Mat image1, Mat image2, std::vector<KeyPoint>* keypointsOut1, std::vector<KeyPoint>* keypointsOut2, int thresh)
{
  if(!keypointsIn1->size() || !keypointsIn2->size())
    {
      //-- Detect the keypoints using SURF Detector
      int minHessian = 400;

      SurfFeatureDetector detector( minHessian );
  
      detector.detect( image1, *keypointsIn1 );
      detector.detect( image2, *keypointsIn2 );
    }

  //-- Extract features for matching
  SurfDescriptorExtractor extractor;

  Mat descriptors1, descriptors2;

  extractor.compute(image1, *keypointsIn1, descriptors1);
  extractor.compute(image2, *keypointsIn2, descriptors2);

  FlannBasedMatcher matcher;
  std::vector<DMatch> matches;

  //-- Match the car SURF points with all the next images SURF points
  matcher.match(descriptors1,descriptors2, matches);

  //-- Find match strengths
  double maxDist = 0; double minDist = 100;

  for( int i = 0; i < descriptors1.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < minDist ) minDist = dist;
      if( dist > maxDist ) maxDist = dist;
    }
  
  //-- Keep good matches
  std::vector< DMatch > goodMatches;
  
  for( int i = 0; i < descriptors1.rows; i++ )
    { if( matches[i].distance <= max(thresh*minDist, 0.02) )
	{ goodMatches.push_back( matches[i]); }
    }

  //-- Fill the good matches keypoint vectors
  for ( int i = 0; i < goodMatches.size(); i++)
    {
      (*keypointsOut1).push_back((*keypointsIn1)[goodMatches[i].queryIdx]);     
      (*keypointsOut2).push_back((*keypointsIn2)[goodMatches[i].trainIdx]);
    }

}

void Car::updateBoxPos(Mat image1, Mat image2)
{

  //-- Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints1, keypoints2, keypointsCar1, keypointsGM1, keypointsGM2;

  detector.detect( image1, keypoints1 );
  detector.detect( image2, keypoints2 );

  Mat imgKeypoints1; Mat imgKeypoints2;
  
  //-- Detect keypoints that are within the box 
 
  for (int i = 0; i < keypoints1.size(); i++)
    {
      if (keypoints1[i].pt.x < lowerRight.x && keypoints1[i].pt.x > upperLeft.x &&
	  keypoints1[i].pt.y < lowerRight.y && keypoints1[i].pt.y > upperLeft.y)
	{
	  keypointsCar1.push_back(keypoints1[i]);
	}

    }
  //-- Match points from within the box to the next frame
  matchPoints(&keypointsCar1,&keypoints2,image1,image2,&keypointsGM1,&keypointsGM2,2.1);

  //-- Use the average distance between the pixel coordinates to generate box movement
  std::vector<double> xDistance, yDistance;
  
  //-- Generate the distances
  for( int i = 0; i < (int)keypointsGM1.size(); i++ )
    { 
      xDistance.push_back(keypointsGM2[i].pt.x - keypointsGM1[i].pt.x);  
      yDistance.push_back(keypointsGM2[i].pt.y - keypointsGM1[i].pt.y); 
    }
  
  //-- The actual average distances in x and y
  double avgX = std::accumulate(xDistance.begin(), xDistance.end(), 0) / (double)xDistance.size();
  double avgY = std::accumulate(yDistance.begin(), yDistance.end(), 0) / (double)yDistance.size();

  drawKeypoints( image1, keypointsGM1, imgKeypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  
  //-- Update the box if we are still tracking the car
  if(!inertialTracking && keypointsGM1.size())
    {
      rectangle(imgKeypoints1,upperLeft,lowerRight,10,3);
      //-- Move the box by the average shift in pixels from this time
      upperLeft.x += avgX;
      lowerRight.x += avgX;
      upperLeft.y += avgY;
      lowerRight.y += avgY;
      xHistory.push_back(avgX);
      yHistory.push_back(avgY);
      //-- Make sure the edges of the box are still on the screen
      checkBounds(image1, &upperLeft, &lowerRight);
    }
  else
    {
      //-- Not sure if the car is in the box
      rectangle(imgKeypoints1,upperLeft,lowerRight,100,5);
      //-- Move the box by the average shift in pixels historically
      double avgXHistorical = std::accumulate(xHistory.begin(), xHistory.end(), 0) / (double)xHistory.size();
      double avgYHistorical = std::accumulate(yHistory.begin(), yHistory.end(), 0) / (double)yHistory.size();

      upperLeft.x += avgXHistorical;
      lowerRight.x += avgXHistorical;
      upperLeft.y += avgYHistorical;
      lowerRight.y += avgYHistorical;
      printf("++++++++++++++++ Inertial Tracking - lost car +++++++++++++++++++\n");
      //-- Make sure the edges of the box are still on the screen
      checkBounds(image1, &upperLeft, &lowerRight);
    }

  //-- Show detected (drawn) keypoints
  imshow("Car Tracker", imgKeypoints1 );
 
  waitKey(10);

}

void Car::getTemplateMatch(Mat image, Point* matchUL, Point* matchLR, Mat* templateImage)
{
  //-- This function will take a template image, resize it to the box size, and find it in the image
  //-- This is used as a check on the SURF features update of the box and the edge resize of the box
 
  //-- If the template image doesn't exist, make one
  if(!templateImage->rows)
    {
      checkBounds(image,&upperLeft,&lowerRight);
      *templateImage = image(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));
    }

  Mat result;
  int resultCols =  image.cols - templateImage->cols + 1;
  int resultRows = image.rows - templateImage->rows + 1;
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  
  result.create( resultCols, resultRows, CV_32FC1 );
      
  //-- Resize the template for the current box size
  Mat resizedTemplate;
  resize(*templateImage,resizedTemplate,Size( lowerRight.x - upperLeft.x,lowerRight.y - upperLeft.y));
  //-- Match and normalize
  matchTemplate(image, resizedTemplate, result, CV_TM_SQDIFF_NORMED );
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
      
  //-- Grab the best match from the result matrix
  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
  *matchUL = minLoc;
    
  //-- Set the other match coordinates
  matchLR->x = matchUL->x + resizedTemplate.cols;
  matchLR->y = matchUL->y + resizedTemplate.rows;
}

void Car::updateBoxSize(Mat image)
{
  //-- This function uses a sparse set of edges to resize the box describing the car
  //-- This assumes there are many edges on the car, and space between the car and edges in the world
  //-- This will drop small edges on the boundaries, to account for edges like the road that are always near the car

  //-- Create an image using Canny edge detection 
  Mat subImage,edges,subEdge;
  
  blur(image, edges, Size(5,5) );
  
  Canny(edges,edges,79,102,3);
  subEdge = edges(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));

  //-- Get the current sum of gradients to compare against
  float currentGradient = sum(subEdge)[0];
  
  //-- This gives us a history of gradient sums to use to check if the current block is similar to the last block
  //-- This did not provide enough information to spend time implementing
  gradientHistory.push_back((int)currentGradient/255);
  double gAvg = std::accumulate(gradientHistory.begin(), gradientHistory.end(), 0) / (double)gradientHistory.size();
  
  //-- Expand the box and see if the increase in captured edges is large enough
  for (int i=0;i<4;i++)
    {
      Point newUL(upperLeft.x-2*i,upperLeft.y-2*i);
      Point newLR(lowerRight.x+2*i,lowerRight.y+2*i);
      checkBounds(edges,&newUL,&newLR);
      subEdge =  edges(Range(newUL.y,newLR.y),Range(newUL.x,newLR.x));
      //-- If there are enough extra edges, keep the new box size
      if (sum(subEdge)[0]/255 > currentGradient/255 + 25)
	{
	  upperLeft = newUL;
	  lowerRight = newLR;
	}
    }

  //-- The box has been greedily expanded, now contract it until there is more than noise on the edges
  int noiseThreshold = 8;
  while(sum(edges(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x-1,upperLeft.x+1)))[0]/255 < noiseThreshold &&
	upperLeft.x < lowerRight.x - 50)
    {
      upperLeft.x += 1;
    }
 
  while(sum(edges(Range(upperLeft.y-1,upperLeft.y+1),Range(upperLeft.x,lowerRight.x)))[0]/255 < noiseThreshold &&
	upperLeft.y < lowerRight.y - 30)
    {
      upperLeft.y += 1;
    }

  while(sum(edges(Range(upperLeft.y,lowerRight.y),Range(lowerRight.x-1,lowerRight.x+1)))[0]/255 < noiseThreshold &&
	upperLeft.x < lowerRight.x - 50)
    {
      lowerRight.x -= 1;
    }

  while(sum(edges(Range(lowerRight.y-1,lowerRight.y+1),Range(upperLeft.x,lowerRight.x)))[0]/255 < noiseThreshold &&
	upperLeft.y < lowerRight.y - 30) 
    {
      lowerRight.y -= 1;
    }

  
  checkBounds(edges,&upperLeft,&lowerRight);    
  rectangle(edges,upperLeft,lowerRight,150);  

  Point matchUL, matchLR;    
 
  //-- Use the current template image to search for the car in the image
  getTemplateMatch(image,&matchUL,&matchLR,&templateIm);
 
  checkBounds(image, &matchUL, &matchLR);    
 
  rectangle( edges, matchUL, matchLR, 100, 2, 8, 0 );

  //-- Allow for error in template matching, up to the overlap factor
  double xOverlap,yOverlap,overlapFactor;
  //-- This will consider success anything more than 50% overlap
  overlapFactor = 0.5;
  xOverlap = (lowerRight.x - upperLeft.x)*overlapFactor;
  yOverlap = (lowerRight.y - upperLeft.y)*overlapFactor;

  if((matchUL.x < upperLeft.x + xOverlap &&
      matchUL.x > upperLeft.x - xOverlap &&
      matchUL.y < upperLeft.y + yOverlap &&
      matchUL.y > upperLeft.y - yOverlap) )
    {
      //-- The template matches the current box and it has been 20 frames since we updated the template
      if(updateTemplate==15)
	{
	  templateIm = image(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));     updateTemplate = 0;
	  printf("------------ Updated template --------------\n");
	}
      else updateTemplate++;
      //-- We are tracking the car
      inertialTracking = false;
    }
  else
    {
      //-- The template isn't matching the box
      inertialTracking = true;
    }

  imshow("Canny Edge Detector",edges);
  waitKey(10);
}

void Car::checkBounds(Mat image, Point* uL, Point* lR)
{
  //If any point doesn't exist, set it to 0
  if(!uL->x) uL->x = 0;
  if(!uL->y) uL->y = 0;
  if(!lR->x) lR->x = 0;
  if(!lR->y) lR->y = 0;

  //Fix the box points if they are outside of bounds
  //The smallest box is 50,30
  if (uL->x < 1) uL->x = 1;
  if (uL->x > image.size().width) uL->x = image.size().width - 52;
  if (lR->x < uL->x + 50) lR->x = uL->x + 50;  
  if (lR->x > image.size().width) lR->x = image.size().width - 1;
  
    
  if (uL->y < 1) uL->y = 1;
  if (uL->y > image.size().height) uL->y = image.size().height - 32;
  if (lR->y < uL->y + 30) lR->y = uL->y + 30;  
  if (lR->y > image.size().height) lR->y = image.size().height - 1;
  
}


bool Car::unitTest(Mat image)
{
  bool allTests = true;
  //-- This function is a simple example of unit testing
  //-- There are frameworks that do this better

  Point matchUL,matchLR;
  
  //-- First test checkBounds
  //-- This function is a check used throughout the class
  checkBounds(image, &matchUL, &matchLR);
  if(matchUL.x > 0 && matchUL.x < image.size().width &&
     matchUL.y > 0 && matchUL.y < image.size().height &&
     matchLR.x > 0 && matchLR.x < image.size().width &&
     matchLR.y > 0 && matchLR.y < image.size().height) printf("[ PASS ]");
  else 
    {
      printf("[ FAIL ]");
      allTests = false;
    }
  printf("  checkBounds Empty Points\n");

  matchUL.x = 0;
  matchUL.y = 0;
  matchLR.x = 0;
  matchLR.y = 0;

  checkBounds(image, &matchUL, &matchLR);
  if(matchUL.x > 0 && matchUL.x < image.size().width &&
     matchUL.y > 0 && matchUL.y < image.size().height &&
     matchLR.x > 0 && matchLR.x < image.size().width &&
     matchLR.y > 0 && matchLR.y < image.size().height) printf("[ PASS ]");
  else 
    {
      printf("[ FAIL ]"); 
      allTests = false;
    }
  printf("  checkBounds Zero Points\n");

  matchUL.x = image.size().width*2;
  matchUL.y = image.size().height*2;
  matchLR.x = image.size().width*2;
  matchLR.y = image.size().height*2;

  checkBounds(image, &matchUL, &matchLR);
  if(matchUL.x > 0 && matchUL.x < image.size().width &&
     matchUL.y > 0 && matchUL.y < image.size().height &&
     matchLR.x > 0 && matchLR.x < image.size().width &&
     matchLR.y > 0 && matchLR.y < image.size().height) printf("[ PASS ]");
  else 
    {
      printf("[ FAIL ]");
      allTests = false;
    }
  printf("  checkBounds Out of Bounds Points\n");

  matchUL.x = image.size().width - 10;
  matchUL.y = image.size().height - 10;
  matchLR.x = 2;
  matchLR.y = 2;

  checkBounds(image, &matchUL, &matchLR);
  if(matchUL.x > 0 && matchUL.x < image.size().width &&
     matchUL.y > 0 && matchUL.y < image.size().height &&
     matchLR.x > 0 && matchLR.x < image.size().width &&
     matchLR.y > 0 && matchLR.y < image.size().height) printf("[ PASS ]");
  else 
    {
      printf("[ FAIL ]");
      allTests = false;
    }
  printf("  checkBounds Reversed Points\n");

  if(templateIm.rows) printf("[ PASS ]");
  else 
    {
      printf("[ FAIL ]"); 
      allTests = false; 
    }
  printf("  Constructor Template Doesn't Exist\n");

  //-- Test getTemplateMatch
  Mat emptyTemplate;
  getTemplateMatch(image,&matchUL,&matchLR,&emptyTemplate);
  if(matchUL== upperLeft) printf("[ PASS ]");
  else 
    {
      printf("[ FAIL ]");
      allTests = false;
    }
  printf("  Empty Template Doesn't Match\n");

  getTemplateMatch(image, &matchUL, &matchLR, &templateIm);
  if(matchUL== upperLeft) printf("[ PASS ]");
  else
    { 
      printf("[ FAIL ]");
      allTests = false;
    }
  printf("  Constructor Template Doesn't Match\n");

  //-- Test matchPoints
  std::vector<KeyPoint> i1,i2,o1,o2;
  matchPoints(&i1,&i2,image,image,&o1,&o2,5);
  if(o1.size()) printf("[ PASS ]");
  else
    {
      printf("[ FAIL ]");
      allTests = false;
    }
  printf("  matchPoints Empty Keypoints\n");

  if (o1[0].pt.x == o2[0].pt.x && o1[0].pt.y == o2[0].pt.y) printf("[ PASS ]");
  else
    {
      printf("[ FAIL ]");
      allTests = false;
    }
  printf("  matchPoints Detector Identical Image\n");
  //-- Test updateBoxPos
  matchUL = upperLeft;
  matchLR = lowerRight;
  updateBoxPos(image,image);
  if(matchUL.x == upperLeft.x) printf("[ PASS ]");
  else
    {
      printf("[ FAIL ]");
      allTests = false;
    }
  printf("  updateBoxPos Identical Image 0 Movement\n");

  //-- Test updateBoxSize - check that the box is about the same size
  matchUL = upperLeft;
  matchLR = lowerRight;
  updateBoxSize(image);
  if(matchUL.x < upperLeft.x + 5 &&
     matchUL.x > upperLeft.x - 5 &&
     matchUL.y < upperLeft.y + 5 &&
     matchUL.y > upperLeft.y - 5 &&
     matchLR.x < lowerRight.x + 5 &&
     matchLR.x > lowerRight.x - 5 &&
     matchLR.x < lowerRight.x + 5 &&
     matchLR.x > lowerRight.x - 5) printf("[ PASS ]");
  else
    {
      printf("[ FAIL ]");
      allTests = false;
    }
  printf("  updateBoxSize Identical Image\n");

  return allTests;
}
