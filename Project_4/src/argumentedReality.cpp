#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

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

// Helper function to create 3D chessboard corners in the target's coordinate space
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
        cv::Point3f(0, 0, -0.1f)  // Z-axis (10 cm along Z, towards the camera)
    };

    // Project the 3D points to the 2D image plane
    std::vector<cv::Point2f> image_points;
    cv::projectPoints(axes_points, rotation_vector, translation_vector, camera_matrix, distortion_coefficients, image_points);

    // Draw the 3D axes on the image
    cv::line(image, image_points[0], image_points[1], cv::Scalar(0, 0, 255), 2); // X-axis in red
    cv::line(image, image_points[0], image_points[2], cv::Scalar(0, 255, 0), 2); // Y-axis in green
    cv::line(image, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 2); // Z-axis in blue
}

int main() {
    // Load camera calibration parameters
    cv::Mat camera_matrix, distortion_coefficients;
    if (!loadCalibrationData("camera_intrinsics.csv", camera_matrix, distortion_coefficients)) {
        return -1;
    }

    // Open video capture (default camera)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    // Chessboard pattern and square size (in meters or arbitrary units)
    const int board_width = 9;
    const int board_height = 6;
    const float square_size = 0.025f; // Adjust based on actual checkerboard square size
    cv::Size patternSize(board_width, board_height);
    std::vector<cv::Vec3f> object_points = create3DChessboardCorners(board_height, board_width, square_size);

    while (true) {
        cv::Mat frame, gray;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        // Convert frame to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Detect chessboard corners
        std::vector<cv::Point2f> corner_set;
        bool found = cv::findChessboardCorners(gray, patternSize, corner_set,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // Refine corner locations
            cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

            // Solve PnP to find the pose of the board relative to the camera
            cv::Mat rotation_vector, translation_vector;
            bool success = cv::solvePnP(object_points, corner_set, camera_matrix, distortion_coefficients,
                                        rotation_vector, translation_vector);

            if (success) {
                // Display rotation and translation vectors
                std::cout << "Rotation Vector:\n" << rotation_vector << std::endl;
                std::cout << "Translation Vector:\n" << translation_vector << std::endl;

                // Convert rotation vector to rotation matrix for display purposes
                cv::Mat rotation_matrix;
                cv::Rodrigues(rotation_vector, rotation_matrix);
                std::cout << "Rotation Matrix:\n" << rotation_matrix << std::endl;

                 // Draw 3D axes on the image
                draw3DAxes(frame, camera_matrix, distortion_coefficients, rotation_vector, translation_vector);

                // Draw the corners
                cv::drawChessboardCorners(frame, patternSize, cv::Mat(corner_set), found);
            }
        }

        // Show the frame
        cv::imshow("Camera Position with 3D Axes", frame);

        // Exit on 'q' key press
        char key = (char)cv::waitKey(30);
        if (key == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
