#ifndef MULTI_HISTOGRAM_MATCHING_H
#define MULTI_HISTOGRAM_MATCHING_H
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream> 
#include <opencv2/opencv.hpp>


// Extract 3D Color Histogram of the top-left of the image
cv::Mat extractMultiHistogram1(cv::Mat &image);

// Extract grayscale Histogram of the image
cv::Mat extractMultiHistogram2(cv::Mat &image);

// Extract 3D Color Histogram of the image and write into a cvs file
void extractAndSaveBothHistogram(const std::string &imagePath, std::ofstream &csvFile);

double computeHistogramIntersection1D(const cv::Mat &histogram1, const cv::Mat &histogram2);


#endif