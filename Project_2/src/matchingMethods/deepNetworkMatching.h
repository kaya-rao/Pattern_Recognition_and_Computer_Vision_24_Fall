#ifndef DNN_MATCHING_H
#define DNN_MATCHING_H
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream> 
#include <opencv2/opencv.hpp>
#include <filesystem>


// compute the SSD between two images features
double computeDnnSSD(const std::vector<float> &featureVector1, const std::vector<float> &featureVector2);

// Extract the filename from a path, 
// use to clean and match the image path for dnn features
std::string extractFilename(const std::string& path);

#endif