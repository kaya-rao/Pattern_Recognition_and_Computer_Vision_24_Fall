#ifndef BASELINE_MATCHING_H
#define BASELINE_MATCHING_H
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream> 
#include <opencv2/opencv.hpp>

// Extract the 7 x 7 pixels from the center
std::vector<int> extractFeatureVector(cv::Mat &image);

// Extract the 7 x 7 pixels from the center and write into a cvs file
void extractAndSaveFeatures(const std::string &imagePath, std::ofstream &csvFile);

// compute the SSD between two images based on the 7 x 7 pixels extracted
double computeSSD(const std::vector<int> &featureVector1, const std::vector<int> &featureVector2);

#endif