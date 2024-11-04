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
#endif 
