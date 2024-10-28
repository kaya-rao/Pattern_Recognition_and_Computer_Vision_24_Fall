/*
  Yunxuan 'Kaya' Rao
  10/27/2024
The collections of the helper functions that's going to apply to the image/live stream
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include "imgProcessing.h"

// Task 1: k means thresholding
int kmeansThreshold(const cv::Mat& src, int K) {
    
    // Convert grayscale image to a single row of data
    cv::Mat samples(src.rows * src.cols, 1, CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            samples.at<float>(y + x * src.rows, 0) = static_cast<float>(src.at<uchar>(y, x));
        }
    }

    // K-means clustering
    cv::Mat labels;
    cv::Mat centers;
    kmeans(samples, K, labels, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
    
    float mean1 = centers.at<float>(0, 0);
    float mean2 = centers.at<float>(1, 0);

    // Set threshold between two means
    return static_cast<int>((mean1 + mean2) / 2);  
}


// Generate a color palette with three colors: red, green, and yellow
std::vector<cv::Vec3b> generateColorPalette() {
    std::vector<cv::Vec3b> palette;
    palette.push_back(cv::Vec3b(0, 0, 255));   // Red
    palette.push_back(cv::Vec3b(0, 255, 0));   // Green
    palette.push_back(cv::Vec3b(0, 255, 255)); // Yellow
    return palette;
}