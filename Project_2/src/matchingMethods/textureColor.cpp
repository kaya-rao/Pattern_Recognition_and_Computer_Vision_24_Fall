#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream> 
#include <opencv2/opencv.hpp>
#include "textureColor.h"


// Extract whole image (3DRGB) Color Histogram of the image
RGBHistogram extractWholeColorHistogram(cv::Mat &image){

    // Define the size of the bins
    int binNum = 8;
    int binSize = 256 / binNum;

    // Define the histogram (Use 3 vectors for easier write/read with csv files)
    // Opencv: BGR
    RGBHistogram histogram;
    histogram.blueHist.resize(binNum);
    histogram.greenHist.resize(binNum);
    histogram.redHist.resize(binNum);

    std::vector<int> bHist(binNum, 0);
    std::vector<int> gHist(binNum, 0);
    std::vector<int> rHist(binNum, 0);
    
    // Go through all the pixels
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
            
            int blueVal = pixel[0];
            int greenVal = pixel[1];
            int redVal = pixel[2];
            // Update the count in histograms
            bHist[blueVal / binSize]++;
            gHist[greenVal / binSize]++;
            rHist[redVal / binSize]++;
        }
    }

    // Normalization
    int totalPixels = image.rows * image.cols;
    for (int i = 0; i < binNum; i++){
        histogram.blueHist[i] = static_cast<float>(bHist[i]) / totalPixels;
        histogram.greenHist[i] = static_cast<float>(gHist[i]) / totalPixels;
        histogram.redHist[i] = static_cast<float>(rHist[i]) / totalPixels;
    }

    return histogram;
}


// Extract whole image (3DRGB) Color Histogram and Texture histogram of the image
void extractAndSaveColorTextureHistogram(const std::string &imagePath, std::ofstream &csvFile){
    // read image
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()){
        std::cerr << "Error reading image: " << imagePath << std::endl;
        return;
    }

    // ------------- RGB Histogram ------------- // 
    RGBHistogram histogram1 = extractWholeColorHistogram(image);
    int binNum1 = 8;
    // Write the data into CSV file
    csvFile << imagePath;
    for (int i = 0; i < binNum1; i++) {
        csvFile << "," << histogram1.blueHist[i];
        csvFile << "," << histogram1.greenHist[i];
        csvFile << "," << histogram1.redHist[i];
    }

    /// ------------- Texture Histogram ------------- // 
    std::vector<float> histogram2 = extractTextureHistogram(image);
    int binNum2 = 16;
    // Write the data into CSV file    
    for (int i = 0; i < binNum2; i++) {
        csvFile << "," << histogram2[i];
    } 

    csvFile << "\n";
}

// horizontal（X）sobel filter - posotive right
int applySobelX3x3( cv::Mat &src, cv::Mat &dst ){
    // Scales, calculates absolute values, and converts the result to 8-bit.
    // cv::convertScaleAbs(tmp, dst); is moved to vidDisplay.cpp
    // so the magnitude and embossingEffect can use CV_16SC3 format data directly
    dst.create(src.size(), CV_16SC3);
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);
    
    // Horizontal
    for(int r = 0; r < src.rows; r++) {
        for(int c = 1; c < src.cols - 1; c++) {
            for(int i = 0; i < 3; i++){
                temp.at<cv::Vec3s>(r, c)[i] = - src.at<cv::Vec3b>(r, c - 1)[i] + src.at<cv::Vec3b>(r, c + 1)[i];
            }
        }
    }

    // Vertical
    for(int r = 1; r < src.rows - 1; r++) {
        for(int c = 0; c < src.cols; c++) {
            for(int i = 0; i < 3; i++){
                dst.at<cv::Vec3s>(r, c)[i] = temp.at<cv::Vec3s>(r - 1, c)[i] + 2 * temp.at<cv::Vec3s>(r, c)[i] + temp.at<cv::Vec3s>(r + 1, c)[i];;
            }
        }
    }
    return(0);
}

// Vertical(Y) 
int applySobelY3x3( cv::Mat &src, cv::Mat &dst ){
    //cv::Mat tmp = cv::Mat::zeros(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);
    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    // Vertical
    for (int r = 1; r < src.rows - 1; r++) {
        for (int c = 0; c < src.cols; c++) {
            for (int i = 0; i < 3; i++) {
                temp.at<cv::Vec3s>(r, c)[i] = src.at<cv::Vec3b>(r - 1, c)[i] - src.at<cv::Vec3b>(r + 1, c)[i];
            }
        }
    }

    // Horizontal
    for (int r = 0; r < src.rows; r++) {
        for (int c = 1; c < src.cols - 1; c++) {  // Avoiding edges
            for (int i = 0; i < 3; i++) {
                dst.at<cv::Vec3s>(r, c)[i] = temp.at<cv::Vec3s>(r, c - 1)[i] + 2 * temp.at<cv::Vec3s>(r, c)[i] + temp.at<cv::Vec3s>(r, c + 1)[i];
            }
        }
    }

    return(0);
}

int calculateMagnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
    // Convert the type of sx and sy
    cv::Mat sxFloat, syFloat;
    sx.convertTo(sxFloat, CV_32F);
    sy.convertTo(syFloat, CV_32F);

    dst.create(sxFloat.size(), CV_32F);

    // magnitude I = sqrt(sx * sx + sy * sy )
    for (int r = 0; r < sxFloat.rows; r++) {
        for (int c = 0; c < sxFloat.cols; c++) {
            // Access the floating-point values for each pixel
            cv::Vec3f gradientX = sxFloat.at<cv::Vec3f>(r, c);
            cv::Vec3f gradientY = syFloat.at<cv::Vec3f>(r, c);

            // Calculate the magnitude for each channel (R, G, B)
            cv::Vec3f magnitude;
            for (int i = 0; i < 3; i++) {
                magnitude[i] = std::sqrt(gradientX[i] * gradientX[i] + gradientY[i] * gradientY[i]);
            }

            // Store the magnitude in the dst matrix
            dst.at<cv::Vec3f>(r, c) = magnitude;
        }
    }
    return(0);
}


// Extract Texture Histogram of the image
std::vector<float> extractTextureHistogram(cv::Mat &image){
    int binNum = 16;
    // Convert image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Calculate Sobel gradients in x and y directions
    cv::Mat gradX, gradY;
    applySobelX3x3(gray, gradX);
    applySobelY3x3(gray, gradY);

    // Calculate gradient magnitude
    cv::Mat magnitude;
    calculateMagnitude(gradX, gradY, magnitude);

    // Find the max value in the magnitude image for normalization
    double minVal, maxVal;
    cv::minMaxLoc(magnitude, &minVal, &maxVal);


    // Bin size based on max value
    float binSize = maxVal / binNum;
    std::vector<float> histogram(binNum, 0);

    // Fill the histogram
    for (int row = 0; row < magnitude.rows; ++row) {
        for (int col = 0; col < magnitude.cols; ++col) {
            float magValue = magnitude.at<float>(row, col);
            int binIndex = std::min(static_cast<int>(magValue / binSize), binNum - 1);
            histogram[binIndex]++;
        }
    }
    // Normalize the histogram
    float totalPixels = static_cast<float>(magnitude.rows * magnitude.cols);
    for (int i = 0; i < binNum; ++i) {
        histogram[i] /= totalPixels;
    }

    // Return the texture histogram
    return {histogram};
}


// Calculate the intersection for a single channel
double computeSingleChannelIntersection(const std::vector<float> &hist1, const std::vector<float> &hist2) {
    double intersection = 0.0;
    if (hist1.size() != hist2.size()) {
        std::cerr << "Histogram size mismatch!" << std::endl;
        return intersection;
    }
    for (size_t i = 0; i < hist1.size(); ++i) {
        intersection += std::min(hist1[i], hist2[i]);
        //std::cout << hist1[i] <<  hist2[i] << std::endl;
    }
    return intersection;
}

// Calculate the intersection for all three channel
double computeHistogramIntersection3D(const RGBHistogram &histogram1, const RGBHistogram &histogram2){
    
    double blueIntersection = computeSingleChannelIntersection(histogram1.blueHist, histogram2.blueHist);
    double greenIntersection = computeSingleChannelIntersection(histogram1.greenHist, histogram2.greenHist);
    double redIntersection = computeSingleChannelIntersection(histogram1.redHist, histogram2.redHist);

    // Normalization with even weight
    double totalIntersection = (blueIntersection + greenIntersection + redIntersection) / 3;
    return -totalIntersection; // Because the bigger the value, the closer it is
}
