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
    bool use_video_capture = true;
    cv::Mat frame;

    // Check if an image file path is provided
    if (argc > 1) {
        std::string input_path = argv[1];
        frame = cv::imread(input_path);

        // If the image could not be loaded, fall back to live video
        if (frame.empty()) {
            printf("Failed to open image. Switching to live video mode.\n");

            // Open live video device
            capdev = new cv::VideoCapture(0);
            if (!capdev->isOpened()) {
                printf("Unable to open video device\n");
                return -1;
            }

            use_video_capture = true;
        } else {
            use_video_capture = false; // Set flag to use the static image
        }
    } else {
        // No file path provided, default to live video
        capdev = new cv::VideoCapture(0);
        if (!capdev->isOpened()) {
            printf("Unable to open video device\n");
            return -1;
        }
    }

    // get some properties of the image
    cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                    (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);
    

    // image counts incase there's are multiply images to save
    // int imgCnt = 0;
    const int min_calibration_images = 5;
    

    // Declare the Frames here
    cv::Mat grayscaleFrame;
    cv::Mat validFrame;

    // Number of inner corners per a chessboard row and column
    const int board_width = 9;
    const int board_height = 6;
    cv::Size patternSize(board_width, board_height);

    // the 3D positions of the corners in world coordinates
    std::vector<cv::Vec3f> point_set; 
	// list of point_set
    std::vector<std::vector<cv::Vec3f> > point_list; 
    // list of 2D corner vectors
	std::vector<std::vector<cv::Point2f> > corner_list; 
    // Images used for calibration
    std::vector<cv::Mat> saved_images;  

    // Camera matrix
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1, 0, capdev->get(cv::CAP_PROP_FRAME_WIDTH) / 2, 0, 1, capdev->get(cv::CAP_PROP_FRAME_HEIGHT) / 2, 0, 0, 1);

    // 5 distortion parameters
    cv::Mat distortion_coefficients = cv::Mat::zeros(5, 1, CV_64F); 
    std::vector<cv::Mat> rotations, translations;
    std::vector<cv::Point2f> corner_set;


    
    // Keep the program running until 'q' input
    while (true) {
        if (use_video_capture) {
            *capdev >> frame;
            if (frame.empty()) {
                printf("Frame is empty\n");
                break;
            }
        }
        // Quit the program if keybaord input = q
        if (cv::waitKey(33) == 'q') break;


        // -------------------------- Task 1: Detect and Extract Target Corners -------------------------- //
        // Convert frame to 8 bit grayscale image
        cv::cvtColor(frame, grayscaleFrame, cv::COLOR_BGR2GRAY);

        // Detect corners
        // std::vector<cv::Point2f> corner_set;
        //CALIB_CB_FAST_CHECK saves a lot of time on images
        //that do not contain any chessboard corners
        bool patternfound = cv::findChessboardCorners(grayscaleFrame, patternSize, corner_set,
                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE
                + cv::CALIB_CB_FAST_CHECK);
        if(patternfound && corner_set.size() == board_width * board_height){ // Only take the corner_set that captured all corners
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
            
            // -------------------------- Task 2: Select Calibration Images -------------------------- //
            // Update the most recent corner_list and image
            validFrame = frame;
        }
    

        // Store vector of the last successfully detected corners if press 's'
        if (cv::waitKey(33) == 's' && corner_set.size() == board_width * board_height){
            // Create a std::vector point_set that specifies the 3D positions of the corners in world coordinates
            point_set = create3DChessboardCorners(board_height, board_width);
            point_list.push_back(point_set);
            saved_images.push_back(validFrame.clone());
            corner_list.push_back(corner_set); 
            std::cout << "Calibration image saved. Total saved: " << corner_list.size() << std::endl;

            // print out the reprojection error if enough data has been collected
            if (corner_list.size() >= min_calibration_images) {
                double reprojection_error = calibrateCameraSystem(camera_matrix, distortion_coefficients, rotations, translations, frame.size(), corner_list, min_calibration_images, point_list);
                std::cout << "Current Reprojection Error: " << reprojection_error << " pixels" << std::endl;
            }

        };

        // -------------------------- Task 3: Calibrate the Camera -------------------------- //
        if (cv::waitKey(33) == 'c'){
            double final_error = calibrateCameraSystem(camera_matrix, distortion_coefficients, rotations, translations, frame.size(), corner_list, min_calibration_images, point_list);
            std::cout << "Final Calibration Error: " << final_error << " pixels" << std::endl;

            // write intrinsic parameters to a csv file
            writeCalibrationToCSV(camera_matrix, distortion_coefficients);

        }

        // Display the image after processing
        cv::namedWindow("Chessboard Corners", cv::WINDOW_NORMAL); 
        cv::resizeWindow("Chessboard Corners", 1000, 550);
        cv::imshow("Chessboard Corners", frame);  
       
    
    }

    delete capdev;
    cv::destroyAllWindows();
    return(0);
}
