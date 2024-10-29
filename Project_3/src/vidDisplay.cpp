/*
  Yunxuan 'Kaya' Rao
  10/22/2024
  Display live stream video and apply filters on it.
 */
#include <cstdio>  // gives me printf
#include <cstring> // gives me strcpy
#include "opencv2/opencv.hpp" // main OpenCV include file
#include "imgProcessing.h"

int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
            printf("Unable to open video device\n");
            return(-1);
        }

        // Lower the FPS to 10 Frames/s
        double targetFPS = 10.0;
        capdev->set(cv::CAP_PROP_FPS, targetFPS);
        double actualFPS = capdev->get(cv::CAP_PROP_FPS);
        std::cout << "FPS set to: " << actualFPS << std::endl;

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);
        
        
        // identifies windows
        cv::namedWindow("Original", 1); 
        cv::namedWindow("Processed", 1); 

        // image counts incase there's are multiply images to save
        int imgCnt = 0;

        // Declare all the Frames here
        cv::Mat frame;
        cv::Mat threadsholdedFrame;
        cv::Mat morphoFrame;
        cv::Mat processedFrame;
        
        
        // Keep the program running until 'q' input
        while (true) {
            *capdev >> frame; // get a new frame from the camera, treat as a stream
            if( frame.empty() ) {
                printf("frame is empty\n");
                break;
            }

            cv::imshow("Video", frame);          
            // see if there is a waiting keystroke
            /*
            char key = cv::waitKey(1);
            if( key == 'q') {
                
                break;
            }*/

            // -------- Tesk 1: threadsholding --------- //
            // --- Pre-processing --- //
            // 1. Convert to grayscale
            cv::Mat grayFrame, blurredFrame;
            greyscale(frame, grayFrame);
            //cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            
            // 2. Apply Gaussian Blur to smooth the image
            gaussianBlur(grayFrame, blurredFrame);
            //cv::GaussianBlur(grayFrame, blurredFrame, cv::Size(5, 5), 0);
            
            // 3. Dynamically set threshold using k-means algorithm
            int thresholdValue = kmeansThreshold(blurredFrame);
            
            // --- Apply the threshold --- //
            threadshold(blurredFrame, threadsholdedFrame, thresholdValue);
            //cv::threshold(blurredFrame, threadsholdedFrame, thresholdValue, 255, cv::THRESH_BINARY);
            //cv::bitwise_not(threadsholdedFrame, threadsholdedFrame);

            cv::imshow("Threadsholded Video", threadsholdedFrame);


            // -------- Tesk 2: Clean up: Closing (Dilation following Erosion) --------- //
            
            cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                cv::Size(5, 5),
                cv::Point(-1, -1)); // set to center

            // Dilation: Filling gaps
            cv::dilate(threadsholdedFrame, morphoFrame, element);

            // Erosion
            cv::erode(morphoFrame, morphoFrame, element);

            
            // -------- Tesk 3: Segmentation --------- //
            int minRegionSize = 100; // Decides the minimum size of a region
            int maxRegionsToShow = 1; // Choose value from 1 to 3
            
            // --- Connected Components Analysis ---//
            cv::Mat labels, stats, centroids;
            // connectivity = 8
            int labelsCnt = connectedComponentsWithStats(morphoFrame, labels, stats, centroids, 4);

            // --- Filter out small regions &  Display regions with different colors --- //
            // Assign colors:
            // Since the maximum number of regions are set to 3
            // It's better to use a fixed set of colors so that
            // different regions' colors are sure to be distinct
            std::vector<cv::Vec3b> colors = generateColorPalette();

            // Create a segmentedFrame with three channels to hold the output colored frame, initializes all pixels to black
            cv::Mat segmentedFrame(morphoFrame.size(), CV_8UC3, cv::Scalar(0, 0, 0));

            int displayedRegions = 0;

            // Loop throught all the labeled regions to filter out smaller ones
            // Start from 1 to skip the background, since background is supposed to be the biggest region
            for (int i = 0; i < labelsCnt && displayedRegions < maxRegionsToShow; ++i) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                int left = stats.at<int>(i, cv::CC_STAT_LEFT);
                int top = stats.at<int>(i, cv::CC_STAT_TOP);
                int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
                int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

                // Check if the area meets the minimum size requirement
                if (area < minRegionSize) continue;

                // Calculate the centroid (the geometric center)
                cv::Point2d centroid(centroids.at<double>(i, 0), centroids.at<double>(i, 1));

                // Check if region is central and does not touch the boundary
                bool isCentral = (centroid.x > width / 2 && centroid.x < morphoFrame.cols - width / 2) && (centroid.y > height / 2 && centroid.y < morphoFrame.rows - height / 2);
                bool touchesBoundary = (left == 0 || top == 0 || left + width == morphoFrame.cols || top + height == morphoFrame.rows);

                if (!isCentral || touchesBoundary) continue;

                // Select color from the colors, assign the colors
                cv::Vec3b color = colors[displayedRegions % 3];
                
                // Color the region
                // Create a mask for the current region
                cv::Mat mask = (labels == i);
                segmentedFrame.setTo(color, mask);

                displayedRegions++;
                
                // -------- Tesk 4: Compute and return features for each major region --------- //
                FeatureVector featureVector = computeAndDisplayRegionFeatures(labels, i, segmentedFrame);

                // -------- Tesk 5: Collect training data --------- //
                if (cv::waitKey(1) == 'n'){
                    std::string label;
                    std::cout << "Enter label for the object: ";
                    std::cin >> label;

                    // Save feature vector with label to a file
                    saveFeatureVector(featureVector, label);
                }
            }

            // Display the result
            cv::imshow("Region Map", segmentedFrame);
            if (cv::waitKey(1) == 'q') break;

        }

        delete capdev;
        cv::destroyAllWindows();
        return(0);
}
