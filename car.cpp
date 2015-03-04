#include "car.h"
using namespace cv;

Car::Car(Point uL, Point lR, Mat initialIm)
{
  upperLeft = uL;
  lowerRight = lR;
  inertialTracking = false;
  templateIm = initialIm(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));
  updateTemplate = 1;  
  /**
     Mat edges;
     blur(initialIm,edges, Size(7,7) );

     Canny(edges,edges,75,92,3);
     templateIm = edges(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));
  
  **/
  
}

Car::~Car()
{
}

void Car::matchPoints(std::vector<KeyPoint>* kp_in1, std::vector<KeyPoint>* kp_in2,Mat image_1, Mat image_2, std::vector<KeyPoint>* kp_out1, std::vector<KeyPoint>* kp_out2, int thresh)
{

  //-- Extract features for matching
  SurfDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor.compute(image_1, *kp_in1, descriptors_1);
  extractor.compute(image_2, *kp_in2, descriptors_2);

  FlannBasedMatcher matcher;
  std::vector<DMatch> matches;

  //-- Match the car SURF points with all the next images SURF points
  matcher.match(descriptors_1,descriptors_2, matches);

  //-- Find match strengths
  double max_dist = 0; double min_dist = 100;

   for( int i = 0; i < descriptors_1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
  
   //-- Keep good matches
   std::vector< DMatch > good_matches;
  
   for( int i = 0; i < descriptors_1.rows; i++ )
     { if( matches[i].distance <= max(thresh*min_dist, 0.02) )
      { good_matches.push_back( matches[i]); }
     }

   //-- Fill the good matches keypoint vectors
   for ( int i = 0; i < good_matches.size(); i++)
     {
       (*kp_out1).push_back((*kp_in1)[good_matches[i].queryIdx]);     
       (*kp_out2).push_back((*kp_in2)[good_matches[i].trainIdx]);
     }

}

void Car::updateBoxPos(Mat image_1, Mat image_2)
{

  //-- Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_car1, keypoints_gm1, keypoints_gm2;

  detector.detect( image_1, keypoints_1 );
  detector.detect( image_2, keypoints_2 );

  Mat img_keypoints_1; Mat img_keypoints_2;
  
  //-- Detect keypoints that are within the box 
 
  for (int i = 0; i < keypoints_1.size(); i++)
    {
      if (keypoints_1[i].pt.x < lowerRight.x && keypoints_1[i].pt.x > upperLeft.x &&
	  keypoints_1[i].pt.y < lowerRight.y && keypoints_1[i].pt.y > upperLeft.y)
	{
	  keypoints_car1.push_back(keypoints_1[i]);
	}

    }
  //-- Match points from within the box to the next frame
  matchPoints(&keypoints_car1,&keypoints_2,image_1,image_2,&keypoints_gm1,&keypoints_gm2,2.1);

  //-- Use the average distance between the pixel coordinates to generate box movement
  std::vector<double> xDistance, yDistance;
  
  //-- Generate the distances
  for( int i = 0; i < (int)keypoints_gm1.size(); i++ )
    { 
      xDistance.push_back(keypoints_gm2[i].pt.x - keypoints_gm1[i].pt.x);  
      yDistance.push_back(keypoints_gm2[i].pt.y - keypoints_gm1[i].pt.y); 
  }
  
  //-- The actual average distances in x and y
  double avgX = std::accumulate(xDistance.begin(), xDistance.end(), 0) / (double)xDistance.size();
  double avgY = std::accumulate(yDistance.begin(), yDistance.end(), 0) / (double)yDistance.size();

  printf( "Distance x: %f  Distance y:  %f  \n", avgX, avgY );
  
  drawKeypoints( image_1, keypoints_gm1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  
  //-- Update the box if we are still tracking the car
  if(!inertialTracking && keypoints_gm1.size())
    {
      rectangle(img_keypoints_1,upperLeft,lowerRight,10,3);
      //-- Move the box by the average shift in pixels from this time
      upperLeft.x += avgX;
      lowerRight.x += avgX;
      upperLeft.y += avgY;
      lowerRight.y += avgY;
      xHistory.push_back(avgX);
      yHistory.push_back(avgY);
      //-- Make sure the edges of the box are still on the screen
      checkBounds(image_1, &upperLeft, &lowerRight);
    }
  else
    {
      //-- Not sure if the car is in the box
      rectangle(img_keypoints_1,upperLeft,lowerRight,100,5);
      //-- Move the box by the average shift in pixels historically
      double avgXHistorical = std::accumulate(xHistory.begin(), xHistory.end(), 0) / (double)xHistory.size();
      double avgYHistorical = std::accumulate(yHistory.begin(), yHistory.end(), 0) / (double)yHistory.size();

      upperLeft.x += avgXHistorical;
      lowerRight.x += avgXHistorical;
      upperLeft.y += avgYHistorical;
      lowerRight.y += avgYHistorical;
      printf("++++++++++++++++ Inertial Tracking - lost car +++++++++++++++++++\n");
      //-- Make sure the edges of the box are still on the screen
      checkBounds(image_1, &upperLeft, &lowerRight);
    }

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );
 
  waitKey(20);

}

void Car::getTemplateMatch(Mat image, Point* matchUL, Point* matchLR)
{
      Mat result;
      int result_cols =  image.cols - templateIm.cols + 1;
      int result_rows = image.rows - templateIm.rows + 1;
      double minVal; double maxVal; Point minLoc; Point maxLoc;
  
      result.create( result_cols, result_rows, CV_32FC1 );

      //-- Do the Matching and Normalize
      
      //-- Resize the template for the current box size
      Mat resizedTemplate;
      resize(templateIm,resizedTemplate,Size( lowerRight.x - upperLeft.x,lowerRight.y - upperLeft.y));
      //-- Use CV_TM_SQDIFF_NORMED or CV_TM_CCORR_NORMED
      matchTemplate(image, resizedTemplate, result, CV_TM_SQDIFF_NORMED );
      normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
      
      minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
      *matchUL = minLoc;
      
      matchLR->x = matchUL->x + resizedTemplate.cols;
      matchLR->y = matchUL->y + resizedTemplate.rows;
}

void Car::updateBoxSize(Mat image)
{
  //-- Create an image using Canny edge detection 
  Mat subImage,edges,subEdge;
  Point extremeUL,extremeLR;
  int expandFactor = 20;
  extremeUL.x = upperLeft.x - expandFactor;
  extremeUL.y = upperLeft.y - expandFactor;
  extremeLR.x = lowerRight.x + expandFactor;
  extremeLR.y = lowerRight.y + expandFactor;
  checkBounds(image,&extremeUL,&extremeLR);
  subImage = image(Range(extremeUL.y,extremeLR.y),Range(extremeUL.x,extremeLR.x));
  blur(image, edges, Size(5,5) );
  
  Canny(edges,edges,70,96,3);
  subEdge = edges(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));

  //-- Get the current gradient to compare against
  float currentGradient = sum(subEdge)[0];
  printf("Current sum: %f\n",currentGradient/255);
  gradientHistory.push_back((int)currentGradient/255);
  double gAvg = std::accumulate(gradientHistory.begin(), gradientHistory.end(), 0) / (double)gradientHistory.size();
  printf("Current avg: %f\n",gAvg);
  
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

 getTemplateMatch(image,&matchUL,&matchLR);
 
 checkBounds(image, &matchUL, &matchLR);    
 
 rectangle( edges, matchUL, matchLR, 100, 2, 8, 0 );

 double xLength,yLength,overlapFactor;
 xLength = (lowerRight.x - upperLeft.x)*overlapFactor;
 yLength = (lowerRight.y - upperLeft.y)*overlapFactor;

 if((matchUL.x < upperLeft.x + xLength &&
     matchUL.x > upperLeft.x - xLength &&
     matchUL.y < upperLeft.y + yLength &&
     matchUL.y > upperLeft.y - yLength) )
   {
     //-- The template matches the current box and its been 5 frames since we updated the template
     if(updateTemplate==20)
       {
	 templateIm = image(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));     updateTemplate = 0;
	 printf("------- Updated template ---------\n");
       }
     else updateTemplate++;
     //-- We are tracking the car
     inertialTracking = false;
     printf("update Template: %d \n", updateTemplate);
   }
 else
   {
     //-- The template isn't matching the box
     inertialTracking = true;
   }


  imshow("gradients",edges);
  waitKey(10);
  }

void Car::checkBounds(Mat image, Point* uL, Point* lR)
{
  //Fix the box points if they are outside of bounds

  if (uL->x < 1) uL->x = 1;
  if (uL->x > image.size().width) uL->x = image.size().width - 4;
  if (lR->x > image.size().width) lR->x = image.size().width - 1;
  if (lR->x < uL->x) lR->x = uL->x + 2;
    
  if (uL->y < 1) uL->y = 1;
  if (uL->y > image.size().height) uL->y = image.size().height - 4;
  if (lR->y > image.size().height) lR->y = image.size().height - 1;
  if (lR->y < uL->y) lR->y = uL->y + 2;
}
