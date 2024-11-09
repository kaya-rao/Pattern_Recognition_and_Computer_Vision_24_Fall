/*
  Yunxuan 'Kaya' Rao
  11/03/2024
The header of the helper functions that's going to apply to the image/live stream
 */
#ifndef IMG_PROCESSING
#define IMG_PROCESSING

#include <opencv2/opencv.hpp>

std::vector<cv::Vec3f> create3DChessboardCorners(int rows, int cols);

double calibrateCameraSystem(cv::Mat &camera_matrix, cv::Mat &distortion_coefficients, std::vector<cv::Mat> &rotations, 
                              std::vector<cv::Mat> &translations, cv::Size image_size, 
                              std::vector<std::vector<cv::Point2f>> &corner_list, int min_calibration_images, std::vector<std::vector<cv::Vec3f>> &point_list);

void writeCalibrationToCSV(const cv::Mat& camera_matrix, const cv::Mat& distortion_coefficients);

// Load calibration parameters from CSV file
bool loadCalibrationData(const std::string& filename, cv::Mat& camera_matrix, cv::Mat& distortion_coefficients);


// Create 3D chessboard corners in the target's coordinate space
std::vector<cv::Vec3f> create3DChessboardCorners(int rows, int cols, float square_size);

// Function to draw 3D axes on the image
void draw3DAxes(cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
                const cv::Mat &rotation_vector, const cv::Mat &translation_vector);


// Draw a small 3D cube on the image
void drawCube(cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
                const cv::Mat &rotation_vector, const cv::Mat &translation_vector);

// Draw a small 3D cube on the image
void drawCubeWithPyramid(cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients,
                const cv::Mat &rotation_vector, const cv::Mat &translation_vector);

// Function to process each frame (image or video) to add Augumented Reality object
void processFrame(cv::Mat &frame, const cv::Mat &camera_matrix, const cv::Mat &distortion_coefficients, bool add_virtual_object, char object_type) ;

#endif 
