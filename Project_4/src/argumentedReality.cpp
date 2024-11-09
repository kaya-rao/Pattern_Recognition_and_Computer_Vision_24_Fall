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

    // Check if a file path is provided (for an image or video)
    if (argc > 1) {
        std::string input_path = argv[1];
        
        // Try to open the file as a video first
        capdev.open(input_path);
        if (!capdev.isOpened()) {
            // If it's not a video, try loading it as an image
            capdev.release();
            cv::Mat static_image = cv::imread(input_path);
            if (static_image.empty()) {
                std::cerr << "Error: Could not open input file (neither video nor image)." << std::endl;
                return -1;
            }
            // Process the single static image
            cv::Mat frame = static_image.clone();
            use_video_capture = false;
            
            // Show the image and allow user to add virtual objects
            processFrame(frame, camera_matrix, distortion_coefficients, false);
            cv::imshow("AR", frame);
            while (true) {
                char key = cv::waitKey(30);
                if (key == 'q') break; // Exit on 'q'
                if (key == 'c' || key == 'p') {
                    // Re-process to add object based on key input
                    processFrame(frame, camera_matrix, distortion_coefficients, true, key);
                    cv::imshow("AR", frame);
                }
            }
        } else {
            // Open live camera capture if no file path is provided
            capdev.open(0);
            if (!capdev.isOpened()) {
                std::cerr << "Error: Could not open the camera." << std::endl;
                return -1;
            }
            use_video_capture = true;
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
            processFrame(frame, camera_matrix, distortion_coefficients, false);
            cv::imshow("AR", frame);
            char key = cv::waitKey(30);
            if (key == 'q') break; // Exit on 'q'
            if (key == 'c' || key == 'p') {
                // Re-process to add object based on key input
                processFrame(frame, camera_matrix, distortion_coefficients, true, key);
                cv::imshow("AR", frame);
            }

        // Exit on 'q' key press
        if (key == 'q') {
            break;
        }
    }

    capdev.release();
    cv::destroyAllWindows();
    return 0;
}
