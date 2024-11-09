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
            cv::Mat static_image = cv::imread(input_path);
            if (static_image.empty()) {
                std::cerr << "Error: Could not open input file (neither video nor image)." << std::endl;
                return -1;
            }
            // Process the single static image
            frame = static_image.clone();
            use_video_capture = false; // Indicate we are using a static image
        }

        while (true){
            // Show the frame and allow user to add virtual objects
            processFrame(frame, camera_matrix, distortion_coefficients, add_virtual_object, object_type);
            cv::imshow("AR", frame);
            char key = cv::waitKey(0);
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
    } else {
        // No file path provided, default to live video
        capdev.open(0);
        if (!capdev.isOpened()) {
            printf("Unable to open video device\n");
            return -1;
        }
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
        char key = cv::waitKey(0);
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
