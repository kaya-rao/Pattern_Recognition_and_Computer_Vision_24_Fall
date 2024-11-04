/*
  Yunxuan 'Kaya' Rao
  11/03/2024
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
    cv::Mat grayscaleFrame;

    // Number of inner corners per a chessboard row and column
    cv::Size patternSize(9, 6);
    
    // Keep the program running until 'q' input
    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if( frame.empty() ) {
            printf("frame is empty\n");
            break;
        }
        // Quit the program if keybaord input = q
        if (cv::waitKey(33) == 'q') break;


        // -------------------------- Task 1: Detect and Extract Target Corners -------------------------- //
        // Convert frame to 8 bit grayscale image
        cv::cvtColor(frame, grayscaleFrame, cv::COLOR_BGR2GRAY);

        // Detect corners
        std::vector<cv::Point2f> corner_set;
        //CALIB_CB_FAST_CHECK saves a lot of time on images
        //that do not contain any chessboard corners
        bool patternfound = cv::findChessboardCorners(grayscaleFrame, patternSize, corner_set,
                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
                + cv::CALIB_CB_FAST_CHECK);
        if(patternfound){
            // Find the corner location
            cv::cornerSubPix(grayscaleFrame, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

            // Draw the corner 
            cv::drawChessboardCorners(frame, patternSize, cv::Mat(corner_set), patternfound);
            
            // Print the number of corners found and the coordinates of the first corner
            std::cout << "Corners found: " << corner_set.size() << std::endl;
            if (!corner_set.empty()) {
                std::cout << "First corner: (" << corner_set[0].x << ", " << corner_set[0].y << ")" << std::endl;
            }
        }
    

        // -------------------------- Task 2: Select Calibration Images -------------------------- //

        // -------------------------- Task 3: Calibrate the Camera -------------------------- //

        // -------------------------- Task 4: Calculate Current Position of the Camera -------------------------- //

        // -------------------------- Task 5: Project Outside Corners or 3D Axes -------------------------- //

        // -------------------------- Task 6: Create a Virtual Object -------------------------- //

        // -------------------------- Task 7: Detect Robust Features -------------------------- //


        // Display the image after processing
        cv::namedWindow("Original Video", cv::WINDOW_NORMAL); 
        cv::resizeWindow("Original Video", 1000, 550);
        cv::imshow("Original Video", frame);  
       
    
    }

    delete capdev;
    cv::destroyAllWindows();
    return(0);
}
