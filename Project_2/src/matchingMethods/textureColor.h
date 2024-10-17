#ifndef TEXTURE_AND_COLOR_MATCHING_H
#define TEXTURE_AND_COLOR_MATCHING_H
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream> 
#include <opencv2/opencv.hpp>

// A Structure to store RGBhistogram
struct RGBHistogram {
    std::vector<float> redHist;
    std::vector<float> greenHist;
    std::vector<float> blueHist;
};

// Extract whole image (3DRGB) Color Histogram of the image
RGBHistogram extractWholeColorHistogram(cv::Mat &image);

// Extract whole image (3DRGB) Color Histogram of the image
//void extractAndSaveWholeColorHistogram(const std::string &imagePath, std::ofstream &csvFile);

// Extract Texture Histogram of the image
std::vector<float> extractTextureHistogram(cv::Mat &image);

// Extract whole image (3DRGB) Color Histogram and Texture histogram of the image
void extractAndSaveColorTextureHistogram(const std::string &imagePath, std::ofstream &csvFile);

// Calculate the intersection for a single channel
double computeSingleChannelIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2);

// Calculate the intersection for all three channel
double computeHistogramIntersection3D(const RGBHistogram &histogram1, const RGBHistogram &histogram2);

#endif