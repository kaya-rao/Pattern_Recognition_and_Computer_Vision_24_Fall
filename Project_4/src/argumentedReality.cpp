/*
  Yunxuan 'Kaya' Rao
  11/03/2024
  Display live stream video and apply filters on it.
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include "imgProcessing.h"

int main(int argc, char** argv) {        
    // Load camera calibration parameters
    cv::Mat camera_matrix, distortion_coefficients;
    if (!loadCalibrationData("camera_intrinsics.csv", camera_matrix, distortion_coefficients)) {
        return -1;
    }
    // Decide the background using
    cv::VideoCapture capdev;
    bool use_video_capture = false;
    cv::Mat frame;
    cv::Mat static_image; // BAM moved this definition here
    
    // Decide if adding an object or not
    bool add_virtual_object = false;
    char object_type = ' ';

    // Check if an input file path is provided
    if (argc > 1) {
        std::string input_path = argv[1];
        std::cout << "Input Path: " << input_path << std::endl;

        // Try to open the file as a video first
        capdev.open(input_path);
        //std::cout<<capdev.isOpened()<<std::endl;
        if (capdev.isOpened()) {
            // If it's not a video, try loading it as an image
            capdev.release();  // Release any previous capture just in case
            static_image = cv::imread(input_path);
            if (static_image.empty()) {
                std::cerr << "Error: Could not open input file (neither video nor image)." << std::endl;
                return -1;
            }
            // Process the single static image
            frame = static_image.clone();
            use_video_capture = false; // Indicate we are using a static image
        }
	printf("Using a static image\n");

        while (true){
            static_image.copyTo(frame); // BAM: always need to start with the initial image so it can find the chessboard
        
            // Show the frame and allow user to add virtual objects
            processFrame(frame, camera_matrix, distortion_coefficients, add_virtual_object, object_type);
            cv::imshow("AR", frame);
            char key = cv::waitKey(15); // BAM: main issue was here, a 0 argument means it waits for a keypress
            if (key == 'q') break; // Exit on 'q'
            if (key == 'c') {
                // Re-process to add object based on key input
                add_virtual_object = true;
                object_type = 'c';
            }
            if (key == 'p') {
                // Re-process to add object based on key input
                add_virtual_object = true;
                object_type = 'p';
            }
            // printf("object_type %c  add_virtual_object %d\n", object_type, (int)add_virtual_object); // BAM debugging
        }
    } else {
        // No file path provided, default to live video
        capdev.open(0);
        if (!capdev.isOpened()) {
            printf("Unable to open video device\n");
            return -1;
        }
        use_video_capture = true;
    }


    while (use_video_capture) {
        cv::Mat frame, gray;
        capdev >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        // Show the frame and allow user to add virtual objects
        processFrame(frame, camera_matrix, distortion_coefficients, add_virtual_object, object_type);
        cv::imshow("AR", frame);
        char key = cv::waitKey(15); // BAM set this to a non-zero value so that it doesn't block the loop
        if (key == 'q') break; // Exit on 'q'
        if (key == 'c') {
            // Re-process to add object based on key input
            add_virtual_object = true;
            object_type = 'c';
        }
        if (key == 'p') {
            // Re-process to add object based on key input
            add_virtual_object = true;
            object_type = 'p';
        }
    }

    capdev.release();
    cv::destroyAllWindows();
    return 0;
}
