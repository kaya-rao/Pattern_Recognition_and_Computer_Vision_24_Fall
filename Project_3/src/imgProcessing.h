/*
  Yunxuan 'Kaya' Rao
  10/27/2024
The header of the helper functions that's going to apply to the image/live stream
 */
#ifndef IMG_PROCESSING
#define IMG_PROCESSING

#include <opencv2/opencv.hpp>

// Task 1: grayscale
// Grayscale filter
int greyscale(cv::Mat &src, cv::Mat &dst);

// Task1: 5 X 5 GaussianBlur blur
int gaussianBlur( cv::Mat &src, cv::Mat &dst);

// Task 1: k means thresholding
int kmeansThreshold(const cv::Mat& input, int K = 2);

void kMeans(const cv::Mat& data, int K, cv::Mat& labels, cv::Mat& centers, int maxIterations = 100);

int threadshold(cv::Mat &src, cv::Mat &dst, int thresholdValue);

// Generate a color palette with three colors: red, green, and yellow
std::vector<cv::Vec3b> generateColorPalette();

void computeAndDisplayRegionFeatures(cv::Mat& regionMap, int regionID, cv::Mat& displayImage);

double euclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2);

#endif // Processings
