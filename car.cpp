#include "car.h"
using namespace cv;

Car::Car(Point uL, Point lR, Mat initialIm)
{
  upperLeft = uL;
  lowerRight = lR;
  inertialTracking = false;
  //templateIm = initialIm(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));

  Mat edges;
  blur(initialIm,edges, Size(7,7) );

  Canny(edges,edges,71,92,3);
  templateIm = edges(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));
  carConfidence = 1;
  
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

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  std::vector< DMatch > good_matches;

  //-- Keep good matches
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

  //-- Add the new points to the confirmed batch
  /*
  if(!confirmedFeatures.size())
    {
      //Mat tempDesc;
      for (int i=0;i<good_matches.size();i++)
	{
	  //tempDesc.row(i) = descriptors_1.row(good_matches[i].queryIdx);
	  confirmedFeatures.push_back(descriptors_1.row(good_matches[i].queryIdx));
	}

    }
  else
    {
      //Check that the new features match with the old ones
     
	  FlannBasedMatcher matcherCheck;
	  std::vector<DMatch> matchesCheck;

	  Mat confFeat(2,confirmedFeatures.size(),confirmedFeatures[0].cols);
	   for (int i=0; i<confirmedFeatures.size();i++)
	{
	  confFeat.row(i) = confirmedFeatures[i];
	}
	  //-- Match the car SURF points with all the next images SURF points
	  matcherCheck.match(confFeat,descriptors_2, matches);

	  //-- Find match strengths
	  double max_dist = 0; double min_dist = 100;

	  for( int j = 0; j < matchesCheck.size(); j++ )
	    { double dist = matchesCheck[j].distance;
	      if( dist < min_dist ) min_dist = dist;
	      if( dist > max_dist ) max_dist = dist;
	    }

	  printf("-- Max dist for confirmed : %f \n",  max_dist );
	  printf("-- Min dist : %f \n", min_dist );

	  std::vector< DMatch > good_matches;

	  //-- Keep good matches
	  for( int i = 0; i < matchesCheck.size(); i++ )
	    { 
	      if( matchesCheck[i].distance <= max(2*min_dist, 0.02) )
		{ good_matches.push_back( matchesCheck[i]); }
	    }
	  printf("Kept matches: %d\n",(int)good_matches.size());

	
	  }*/

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
  
  //-- Detect keypoints that are within the box, and then find matches 
 
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
    //printf( "-- Good Match [%d] Keypoint 1x: %f  -- Keypoint 2x: %f  \n", i, keypoints_gm1[i].pt.x, keypoints_gm2[i].pt.x ); 
    xDistance.push_back(keypoints_gm2[i].pt.x - keypoints_gm1[i].pt.x);  
    yDistance.push_back(keypoints_gm2[i].pt.y - keypoints_gm1[i].pt.y); 
    //printf( "-- Good Match [%d] xDistance: %f  -- yDistance: %f  \n", i, xDistance[i], yDistance[i] ); 
  }
  
  //-- The actual average distances in x and y
  double avgX = std::accumulate(xDistance.begin(), xDistance.end(), 0) / (double)xDistance.size();
  double avgY = std::accumulate(yDistance.begin(), yDistance.end(), 0) / (double)yDistance.size();

  printf( "Distance x: %f  Distance y:  %f  \n", avgX, avgY );

  
  drawKeypoints( image_1, keypoints_gm1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  //drawKeypoints( image_2, keypoints_gm2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  /*
  if(!inertialTracking)
    {
      rectangle(img_keypoints_1,upperLeft,lowerRight,10,3);
    }
  else
    {
      //-- Not sure if the car is in the box
      rectangle(img_keypoints_1,upperLeft,lowerRight,100,5);
    } */

 
  
  
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
    
      //-- Make sure the edges of the box are still on the screen
      checkBounds(image_1, &upperLeft, &lowerRight);
    }

  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );
 
  waitKey(1);

}

void Car::updateBoxSize(Mat image)
{
  Mat edges,subEdge;

  blur(image, edges, Size(7,7) );

  Canny(edges,edges,75,102,3);
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
   while(sum(edges(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x-1,upperLeft.x+1)))[0]/255 < 6 &&
upperLeft.x < lowerRight.x - 50)
     {
       upperLeft.x += 1;
     }
 
 while(sum(edges(Range(upperLeft.y-1,upperLeft.y+1),Range(upperLeft.x,lowerRight.x)))[0]/255 < 6 &&
upperLeft.y < lowerRight.y - 30)
     {
       upperLeft.y += 1;
     }

 while(sum(edges(Range(upperLeft.y,lowerRight.y),Range(lowerRight.x-1,lowerRight.x+1)))[0]/255 < 6 &&
upperLeft.x < lowerRight.x - 50)
     {
       lowerRight.x -= 1;
     }

 while(sum(edges(Range(lowerRight.y-1,lowerRight.y+1),Range(upperLeft.x,lowerRight.x)))[0]/255 < 6 &&
upperLeft.y < lowerRight.y - 30) 
     {
       lowerRight.y -= 1;
     }

   // printf("Sum along ULX: %f\n",sum(edges(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,upperLeft.x+1)))[0]/255);

   // printf("Sum along ULY: %f\n",sum(edges(Range(upperLeft.y,upperLeft.y+1),Range(upperLeft.x,lowerRight.x)))[0]/255);

   // printf("Sum along LRX: %f\n",sum(edges(Range(upperLeft.y,lowerRight.y),Range(lowerRight.x,lowerRight.x+1)))[0]/255);

   // printf("Sum along LRY: %f\n",sum(edges(Range(lowerRight.y,lowerRight.y+1),Range(upperLeft.x,lowerRight.x)))[0]/255);
 checkBounds(edges,&upperLeft,&lowerRight);    
 rectangle(edges,upperLeft,lowerRight,150);  

      //BEGIN TEMPLATE BLOCK
      Mat result;
      int result_cols =  edges.cols - templateIm.cols + 1;
      int result_rows = edges.rows - templateIm.rows + 1;
      printf("#########rcols %d \n",result_cols);
      printf("#########rrows %d \n",result_rows);

      result.create( result_cols, result_rows, CV_32FC1 );

      /// Do the Matching and Normalize
      //matchTemplate( edges, templateIm, result, CV_TM_SQDIFF_NORMED );

      /// Localizing the best match with minMaxLoc
      double minVal; double maxVal; Point minLoc; Point maxLoc;
      Point matchLoc;    
  
      //resize the template for the current box size
     Mat resizedTemplate;
     resize(templateIm,resizedTemplate,Size( lowerRight.x - upperLeft.x,lowerRight.y - upperLeft.y));
      
      matchTemplate(edges, resizedTemplate, result, CV_TM_CCORR_NORMED );
      normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
      
      minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
      matchLoc = maxLoc;
      printf("match amount: %f\n",maxVal);
  
      Point matchMate(matchLoc.x + resizedTemplate.cols , matchLoc.y + resizedTemplate.rows);
      checkBounds(edges, &matchLoc, &matchMate);    

      /** using normal template
      matchTemplate(edges, templateIm, result, CV_TM_CCORR_NORMED );
      normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
   

      minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

      /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
      
      //matchLoc = minLoc; 
      matchLoc = maxLoc; 
  
      Point matchMate(matchLoc.x + templateIm.cols , matchLoc.y + templateIm.rows);
      checkBounds(edges, &matchLoc, &matchMate);

**/
      /// Show me what you got
      rectangle( edges, matchLoc, matchMate, 100, 2, 8, 0 );
  

  if(matchLoc.x < upperLeft.x + 10 &&
     matchLoc.x > upperLeft.x -10 &&
     matchLoc.y < upperLeft.y + 10 &&
     matchLoc.y > upperLeft.y -10 ||
     matchMate.x < lowerRight.x + 10 &&
     matchMate.x > lowerRight.x -10 &&
     matchMate.y < lowerRight.y + 10 &&
     matchMate.y > lowerRight.y -10 )
    {
      //-- The template matches the current box and its been 5 frames since we updated the template
      if(carConfidence==6)
	{
	  templateIm = edges(Range(upperLeft.y,lowerRight.y),Range(upperLeft.x,lowerRight.x));     carConfidence = 0;
	  printf("------- Updated template ---------\n");
	}
      else carConfidence++;
      inertialTracking = false;
    }
  else
    {
      //-- The template isn't matching the box
      inertialTracking = true;
    }

  //END TEMPLATE BLOCK


  imshow("gradients",edges);
  waitKey(1);
  }

void Car::checkBounds(Mat image, Point* uL, Point* lR)
{
  //Fix the box points if they are outside of bounds

  if (uL->x < 1) uL->x = 1;
  if (uL->x > image.size().height) uL->x = image.size().height - 4;
  if (lR->x > image.size().height) lR->x = image.size().height - 1;
  if (lR->x < uL->x) lR->x = uL->x + 2;
    
  if (uL->y < 1) uL->y = 1;
  if (uL->y > image.size().width) uL->y = image.size().width - 4;
  if (lR->y > image.size().width) lR->y = image.size().width - 1;
  if (lR->y < uL->y) lR->y = uL->y + 2;
}
