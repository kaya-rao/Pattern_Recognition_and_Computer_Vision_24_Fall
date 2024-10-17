#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream> 
#include <opencv2/opencv.hpp>
#include "histogramMatching.h"


// Extract 2D Color Histogram of the image
cv::Mat extract2DColorHistogram(cv::Mat &image){
    int BIN_NUM = 16;
    int binSize = 256 / BIN_NUM;
    cv::Mat histogram = cv::Mat::zeros(BIN_NUM, BIN_NUM, CV_32F);
    
    // loop through each pixel to generate the histogram 
    for (int r = 0; r < image.rows; r++){
        cv::Vec3b *srcptr = image.ptr<cv::Vec3b>(r);
        for (int c = 0; c < image.cols; c++){
            int blue = srcptr[c][0];
            int red = srcptr[c][2]; 

            // Locate the bins
            int blueBin = static_cast<int>(blue / 256.0 * binSize);
            int redBin = static_cast<int>(red / 256.0 * binSize);

            // Update histogram
            histogram.at<float>(blueBin, redBin)++;
        }
    }
    // Normalization
    cv::normalize(histogram, histogram, 1, 0, cv::NORM_L1);
    return histogram;
}

// Extract 2D Color Histogram of the image and write into a cvs file
void extractAndSaveHistogram(const std::string &imagePath, std::ofstream &csvFile){
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);  // Read the image in color
    if (image.empty()) {
        std::cerr << "Error reading image: " << imagePath << std::endl;
        return;
    }

    cv::Mat histogram = extract2DColorHistogram(image);

    // Write the image path to the CSV file
    csvFile << imagePath;
    
    // Write the histogram to the CSV file
    for (int r = 0; r < histogram.rows; r++) {
        for (int c = 0; c < histogram.cols; c++) {
            csvFile << "," << histogram.at<float>(r, c);
        }
    }

    csvFile << "\n";
}

double computeHistogramIntersection(const cv::Mat &histogram1, const cv::Mat &histogram2){
    double intersection = 0.0;
    for (int r = 0; r < histogram1.rows; r++) {
        for (int c = 0; c < histogram1.cols; c++) {
            intersection += std::min(histogram1.at<float>(r, c), histogram2.at<float>(r, c));
        }
    }
    return -intersection;  // Higher value means more similarity, therefor putting the opposite value
}