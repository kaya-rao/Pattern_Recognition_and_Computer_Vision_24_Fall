/*
  Yunxuan 'Kaya' Rao
  11/03/2024
The collections of the helper functions that's going to apply to the image/live stream
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include "imgProcessing.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <numeric>



// Create 3D world points for a chessboard pattern
std::vector<cv::Vec3f> create3DChessboardCorners(int rows, int cols)
{
    std::vector<cv::Vec3f> points;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            points.emplace_back(j, -i, 0);
        }
    }
    return points;
}

// Perform camera calibration
double calibrateCameraSystem(cv::Mat &camera_matrix, cv::Mat &distortion_coefficients, std::vector<cv::Mat> &rotations, 
                              std::vector<cv::Mat> &translations, cv::Size image_size, 
                              std::vector<std::vector<cv::Point2f>> &corner_list, int min_calibration_images, std::vector<std::vector<cv::Vec3f>> &point_list) {

    std::vector<cv::Mat> saved_images;  
    if (corner_list.size() < min_calibration_images) {
        std::cerr << "Not enough calibration images. Minimum required: " << min_calibration_images << std::endl;
        return -1.0;
    }

    // calibration
    double reprojection_error = cv::calibrateCamera(point_list, corner_list, image_size, camera_matrix,
                                                    distortion_coefficients, rotations, translations,
                                                    cv::CALIB_FIX_ASPECT_RATIO);

    // Print results
    std::cout << "Camera Matrix:\n" << camera_matrix << std::endl;
    std::cout << "Distortion Coefficients:\n" << distortion_coefficients << std::endl;
    std::cout << "Reprojection Error: " << reprojection_error << " pixels" << std::endl;
    return reprojection_error;
}


// Write camera matrix and distortion coefficients to a CSV file
void writeCalibrationToCSV(const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients) {
    std::ofstream file("camera_intrinsics.csv");

    if (file.is_open()) {
        file << "Camera Matrix," << std::endl;
        for (int i = 0; i < camera_matrix.rows; ++i) {
            for (int j = 0; j < camera_matrix.cols; ++j) {
                file << camera_matrix.at<double>(i, j);
                if (j < camera_matrix.cols - 1) {
                    file << ",";
                }
            }
            file << std::endl;
        }

        file << "Distortion Coefficients," << std::endl;
        for (int i = 0; i < distortion_coefficients.rows; ++i) {
            file << distortion_coefficients.at<double>(i, 0);
            if (i < distortion_coefficients.rows - 1) {
                file << ",";
            }
        }
        file << std::endl;

        file.close();
        std::cout << "Calibration data saved to camera_intrinsics.csv" << std::endl;
    } else {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
    }
}