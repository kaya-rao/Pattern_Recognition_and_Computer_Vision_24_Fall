/*
  Yunxuan 'Kaya' Rao
  10/27/2024
The header of the helper functions that's going to apply to the image/live stream
 */
#ifndef IMG_PROCESSING
#define IMG_PROCESSING

#include <opencv2/opencv.hpp>

// Task 1: k means thresholding
int kmeansThreshold(const cv::Mat& input, int K = 2);

// Generate a color palette with three colors: red, green, and yellow
std::vector<cv::Vec3b> generateColorPalette();

#endif // Processings
