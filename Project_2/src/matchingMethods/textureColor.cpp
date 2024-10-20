#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream> 
#include <opencv2/opencv.hpp>
#include "textureColor.h"


// Extract whole image (3DRGB) Color Histogram of the image
cv::Mat extractWholeColorHistogram(cv::Mat &image){

    // Define the size of the bins
    int binNum = 8;
    int binSize = 256 / binNum;

    // Create the 3D histogram
    int histSize[] = { binNum, binNum, binNum };
    cv::Mat histogram = cv::Mat::zeros(3, histSize, CV_32F);

    // Opencv: BGR    
    // Go through all the pixels
    for (int row = 0; row < image.rows; row++) {
        for (int col = 0; col < image.cols; col++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);
            
            // Get the value
            int blueVal = pixel[0] / binSize;
            int greenVal = pixel[1] / binSize;
            int redVal = pixel[2] / binSize;

            // Update the count in histograms
            histogram.at<float>(blueVal, greenVal, redVal)++;
        }
    }

    // Normalization
    int totalPixels = image.rows * image.cols;
    histogram /= totalPixels;
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

    // Write the filepath to the CSV file
    csvFile << imagePath;
    
    // ------------- RGB Histogram ------------- // 
    cv::Mat histogram1 = extractWholeColorHistogram(image);
    int binNum1 = 8;
    // Write the data into CSV file
    for (int blueVal = 0; blueVal < binNum1; blueVal++) {
        for (int greenVal = 0; greenVal < binNum1; greenVal++) {
            for (int redVal = 0; redVal < binNum1; redVal++) {
                float binValue = histogram1.at<float>(blueVal, greenVal, redVal);
                csvFile << "," << binValue;
            }
        }
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
    // initialize the dst Mat to zeros
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);
    
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

    // initialize the dst Mat to zeros
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    cv::Mat temp = cv::Mat::zeros(src.size(), CV_16SC3);


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
double computeHistogramIntersection3D(const cv::Mat &histogram1, const cv::Mat &histogram2){
    double intersection = 0.0;

    // Ensure both histograms have the same size
    if (histogram1.size != histogram2.size) {
        std::cerr << "Histogram size mismatch!" << std::endl;
        return intersection;
    }

    // Traverse through each bin in the 3D histogram
    int binNum = histogram1.size[0];  // Assuming cubic 3D histogram (e.g., 8x8x8)
    for (int blueBin = 0; blueBin < binNum; blueBin++) {
        for (int greenBin = 0; greenBin < binNum; greenBin++) {
            for (int redBin = 0; redBin < binNum; redBin++) {
                float histValue1 = histogram1.at<float>(blueBin, greenBin, redBin);
                float histValue2 = histogram2.at<float>(blueBin, greenBin, redBin);
                // Compute the intersection
                intersection += std::min(histValue1, histValue2);
            }
        }
    }
    
    return -intersection; // Because the bigger the value, the closer it is
}
