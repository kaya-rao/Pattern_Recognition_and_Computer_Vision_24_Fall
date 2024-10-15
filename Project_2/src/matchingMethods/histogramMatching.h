#ifndef HISTOGRAM_MATCHING_H
#define HISTOGRAM_MATCHING_H
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream> 
#include <opencv2/opencv.hpp>

// Extract 2D Color Histogram of the image
cv::Mat extract2DColorHistogram(cv::Mat &image);

// Extract 2D Color Histogram of the image and write into a cvs file
void extractAndSaveHistogram(const std::string &imagePath, std::ofstream &csvFile);

// compute the SSD between two images based on the 7 x 7 pixels extracted
double computeHistogramIntersection(const cv::Mat &histogram1, const cv::Mat &histogram2);

#endif