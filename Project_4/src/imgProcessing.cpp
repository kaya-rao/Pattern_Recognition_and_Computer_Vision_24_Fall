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
    // wirte to camera_intrinsics.csv with continous writting
    // std::ofstream file("camera_intrinsics.csv", std::ios::app);
    // Only need one set of camera intrinsics, 
    // Keep rewriting the csv file for updating
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


// Load calibration parameters from CSV file
bool loadCalibrationData(const std::string& filename, cv::Mat& camera_matrix, cv::Mat& distortion_coefficients) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open the calibration file." << std::endl;
        return false;
    }

    std::string line;
    // Load Camera Matrix
    std::getline(file, line); // Skip header line
    camera_matrix = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < camera_matrix.rows; ++i) {
        for (int j = 0; j < camera_matrix.cols; ++j) {
            if (file >> camera_matrix.at<double>(i, j)) {
                if (j < camera_matrix.cols - 1) file.ignore(1, ','); // Skip commas
            }
        }
    }

    // Load Distortion Coefficients
    std::getline(file, line); // Skip header line
    distortion_coefficients = cv::Mat(5, 1, CV_64F);
    for (int i = 0; i < distortion_coefficients.rows; ++i) {
        if (file >> distortion_coefficients.at<double>(i, 0)) {
            if (i < distortion_coefficients.rows - 1) file.ignore(1, ','); // Skip commas
        }
    }

    file.close();
    return true;
}

// Create 3D chessboard corners in the target's coordinate space
std::vector<cv::Vec3f> create3DChessboardCorners(int rows, int cols, float square_size) {
    std::vector<cv::Vec3f> points;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            points.emplace_back(j * square_size, -i * square_size, 0);
        }
    }
    return points;
}

// Function to draw 3D axes on the image
void draw3DAxes(cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
                const cv::Mat &rotation_vector, const cv::Mat &translation_vector) {
    // Define the 3D points for the 3D axes (in the target's coordinate space)
    std::vector<cv::Point3f> axes_points = {
        cv::Point3f(0, 0, 0),  // Origin
        cv::Point3f(0.1f, 0, 0),  // X-axis (10 cm along X)
        cv::Point3f(0, 0.1f, 0),  // Y-axis (10 cm along Y)
        cv::Point3f(0, 0, 0.1f)  // Z-axis (10 cm along Z, towards the camera)
    };

    // Project the 3D points to the 2D image plane
    std::vector<cv::Point2f> image_points;
    /*
        void cv::projectPoints	(	InputArray 	objectPoints,
        InputArray 	rvec,
        InputArray 	tvec,
        InputArray 	cameraMatrix,
        InputArray 	distCoeffs,
        OutputArray 	imagePoints,
        OutputArray 	jacobian = noArray(),
        double 	aspectRatio = 0 
        )	
    */
    cv::projectPoints(axes_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients, image_points);

    // Draw the 3D axes on the image
    cv::line(image, image_points[0], image_points[1], cv::Scalar(0, 0, 255), 2); // X-axis in red
    cv::line(image, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 2); // Y-axis in green
    cv::line(image, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 2); // Z-axis in blue

    // Add label
    cv::putText(image, "X", image_points[1], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "Y", image_points[2], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, "Z", image_points[3], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
}


// Draw a small 3D cube on the image
void drawCube(cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
                const cv::Mat &rotation_vector, const cv::Mat &translation_vector) {
    // Define the 3D points for a cube (in the 3D world coordinate space)
    std::vector<cv::Point3f> cube_points = {
        cv::Point3f(0.045f, -0.045f, 0.045f),  // 1
        cv::Point3f(0.090f, -0.045f, 0.045f),  // 2
        cv::Point3f(0.045f, -0.090f, 0.045f),  // 3
        cv::Point3f(0.045f, -0.045f, 0.090f),  // 4
        cv::Point3f(0.090f, -0.045f, 0.090f),  // 5
        cv::Point3f(0.090f, -0.090f, 0.090f),  // 6
        cv::Point3f(0.045f, -0.090f, 0.090f),  // 7
        cv::Point3f(0.090f, -0.090f, 0.045f)   // 8
    };

    // Project the 3D points to the 2D image plane
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(cube_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients, image_points);

    // Draw the cube edges on the image
    cv::line(image, image_points[0], image_points[1], cv::Scalar(203, 20, 255), 2); // 1 - 2
    cv::line(image, image_points[0], image_points[2], cv::Scalar(203, 20, 255), 2); // 1 - 3
    cv::line(image, image_points[0], image_points[3], cv::Scalar(203, 20, 255), 2); // 1 - 4
    cv::line(image, image_points[1], image_points[4], cv::Scalar(203, 20, 255), 2); // 2 - 5
    cv::line(image, image_points[1], image_points[7], cv::Scalar(203, 20, 255), 2); // 2 - 8
    cv::line(image, image_points[2], image_points[6], cv::Scalar(203, 20, 255), 2); // 3 - 7
    cv::line(image, image_points[2], image_points[7], cv::Scalar(203, 20, 255), 2); // 3 - 8
    cv::line(image, image_points[3], image_points[6], cv::Scalar(203, 20, 255), 2); // 4 - 7
    cv::line(image, image_points[4], image_points[5], cv::Scalar(203, 20, 255), 2); // 5 - 6
    cv::line(image, image_points[5], image_points[7], cv::Scalar(203, 20, 255), 2); // 6 - 8
    cv::line(image, image_points[5], image_points[6], cv::Scalar(203, 20, 255), 2); // 6 - 7
    cv::line(image, image_points[3], image_points[4], cv::Scalar(203, 20, 255), 2); // 4 - 5

    // Add label
    cv::putText(image, "1", image_points[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "2", image_points[1], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "3", image_points[2], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "4", image_points[3], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "5", image_points[4], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "6", image_points[5], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "7", image_points[6], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "8", image_points[7], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
}

// Draw a small 3D cube on the image
void drawCubeWithPyramid(cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
                const cv::Mat &rotation_vector, const cv::Mat &translation_vector) {
    // Define the 3D points for a cube (in the 3D world coordinate space)
    std::vector<cv::Point3f> cube_points = {
        cv::Point3f(0.045f + 0.15f, -0.045f, 0.145f),  // 1
        cv::Point3f(0.090f + 0.15f, -0.045f, 0.145f),  // 2
        cv::Point3f(0.045f + 0.15f, -0.090f, 0.145f),  // 3
        cv::Point3f(0.045f + 0.15f, -0.045f, 0.190f),  // 4
        cv::Point3f(0.090f + 0.15f, -0.045f, 0.190f),  // 5
        cv::Point3f(0.090f + 0.15f, -0.090f, 0.190f),  // 6
        cv::Point3f(0.045f + 0.15f, -0.090f, 0.190f),  // 7
        cv::Point3f(0.090f + 0.15f, -0.090f, 0.145f)   // 8
    };

    // Triangular pyramids on the cube's surface
    std::vector<cv::Point3f> pyramid_points = {
        cv::Point3f(0.2175f , -0.03f, 0.1675f),  // Tip of the first pyramid
    };

    // Project the 3D points to the 2D image plane
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(cube_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients, image_points);

    // Draw the cube edges on the image
    cv::line(image, image_points[0], image_points[1], cv::Scalar(203, 20, 255), 2); // 1 - 2
    cv::line(image, image_points[0], image_points[2], cv::Scalar(203, 20, 255), 2); // 1 - 3
    cv::line(image, image_points[0], image_points[3], cv::Scalar(203, 20, 255), 2); // 1 - 4
    cv::line(image, image_points[1], image_points[4], cv::Scalar(203, 20, 255), 2); // 2 - 5
    cv::line(image, image_points[1], image_points[7], cv::Scalar(203, 20, 255), 2); // 2 - 8
    cv::line(image, image_points[2], image_points[6], cv::Scalar(203, 20, 255), 2); // 3 - 7
    cv::line(image, image_points[2], image_points[7], cv::Scalar(203, 20, 255), 2); // 3 - 8
    cv::line(image, image_points[3], image_points[6], cv::Scalar(203, 20, 255), 2); // 4 - 7
    cv::line(image, image_points[4], image_points[5], cv::Scalar(203, 20, 255), 2); // 5 - 6
    cv::line(image, image_points[5], image_points[7], cv::Scalar(203, 20, 255), 2); // 6 - 8
    cv::line(image, image_points[5], image_points[6], cv::Scalar(203, 20, 255), 2); // 6 - 7
    cv::line(image, image_points[3], image_points[4], cv::Scalar(203, 20, 255), 2); // 4 - 5

    // Add label
    cv::putText(image, "1", image_points[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "2", image_points[1], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "3", image_points[2], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "4", image_points[3], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "5", image_points[4], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "6", image_points[5], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "7", image_points[6], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, "8", image_points[7], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);

    // Project the 3D points of the pyramids to the 2D image plane
    std::vector<cv::Point2f> pyramid_image_points;
    cv::projectPoints(pyramid_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients, pyramid_image_points);

    // Draw the first pyramid on the x-y plane
    cv::line(image, image_points[0], pyramid_image_points[0], cv::Scalar(0, 255, 0), 2); // Base to tip of the first pyramid
    cv::line(image, image_points[1], pyramid_image_points[0], cv::Scalar(0, 255, 0), 2); // Base to tip
    cv::line(image, image_points[3], pyramid_image_points[0], cv::Scalar(0, 255, 0), 2); // Base to tip
    cv::line(image, image_points[4], pyramid_image_points[0], cv::Scalar(0, 255, 0), 2); // Base to tip
}


// Function to process each frame (image or video) to add Augumented Reality object
void processFrame(cv::Mat &frame, const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients, bool add_virtual_object, char object_type) {
    // Chessboard pattern and square size 
    const int board_width = 9;
    const int board_height = 6;
    const float square_size = 0.025f; // Checkerboard square size
    cv::Size patternSize(board_width, board_height);
    std::vector<cv::Vec3f> object_points = create3DChessboardCorners(board_height, board_width, square_size);

    // Convert frame to grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // Detect chessboard corners
    std::vector<cv::Point2f> corner_set;
    bool found = cv::findChessboardCorners(gray, patternSize, corner_set,
                                            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (found) {
        // -------------------------- Task 4: Calculate Current Position of the Camera -------------------------- //
        // Refine corner locations
        cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

        // Solve PnP to find the pose of the board relative to the camera
        cv::Mat rotation_vector, translation_vector;
        bool success = cv::solvePnP(object_points, corner_set, camera_matrix, distortion_coefficients, rotation_vector, translation_vector);

        if (success) {
            // Display rotation and translation vectors
            std::cout << "Rotation Vector:\n" << rotation_vector << std::endl;
            std::cout << "Translation Vector:\n" << translation_vector << std::endl;

            // Convert rotation vector to rotation matrix for display purposes
            cv::Mat rotation_matrix;
            cv::Rodrigues(rotation_vector, rotation_matrix);
            std::cout << "Rotation Matrix:\n" << rotation_matrix << std::endl;

            // -------------------------- Task 5: Project Outside Corners and 3D Axes -------------------------- //
            // Draw 3D axes on the image
            draw3DAxes(frame, camera_matrix, distortion_coefficients, rotation_vector, translation_vector);

            // Draw the corners
            cv::drawChessboardCorners(frame, patternSize, cv::Mat(corner_set), found);

            // -------------------------- Task 6: Create a Virtual Object (if requested) -------------------------- //
            if (add_virtual_object) {
                if (object_type == 'c') {
                    drawCube(frame, camera_matrix, distortion_coefficients, rotation_vector, translation_vector);
                } 
                if (object_type == 'p') {
                    drawCubeWithPyramid(frame, camera_matrix, distortion_coefficients, rotation_vector, translation_vector);
                }
            }
        }
    }
}
