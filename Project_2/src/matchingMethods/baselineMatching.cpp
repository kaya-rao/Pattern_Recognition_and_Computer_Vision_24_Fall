#include "baselineMatching.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream> 


// Extract the 7 x 7 pixels from the center and write into a cvs file
// Keep the color channels instead of converting to grayscale

std::vector<int> extractFeatureVector(cv::Mat &image) {
    std::vector<int> featureVector;
    int w = image.cols;
    int h = image.rows;
    int startX = w / 2 - 3;
    int startY = h / 2 - 3;

    cv::Mat featurePatch = image(cv::Rect(startX, startY, 7, 7));
    
    for (int i = 0; i < featurePatch.rows; i++) {
        for (int j = 0; j < featurePatch.cols; j++) {
            cv::Vec3b pixel = featurePatch.at<cv::Vec3b>(i, j);
            int blue = pixel[0];
            int green = pixel[1];
            int red = pixel[2];

            // Add BGR values to the feature vector
            featureVector.push_back(blue);
            featureVector.push_back(green);
            featureVector.push_back(red);
        }
    }
    return featureVector;
}

// Function to extract 7x7 feature with color information from an image and save it to CSV
void extractAndSaveFeatures(const std::string &imagePath, std::ofstream &csvFile) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);  // Read the image in color
    if (image.empty()) {
        std::cerr << "Error reading image: " << imagePath << std::endl;
        return;
    }

    // Get the center 7x7 square
    int w = image.cols;
    int h = image.rows;
    int startX = w / 2 - 3;
    int startY = h / 2 - 3;

    cv::Mat featurePatch = image(cv::Rect(startX, startY, 7, 7));  // Extract the 7x7 patch

    // Flatten and store in CSV file
    csvFile << imagePath;
    
    for (int i = 0; i < featurePatch.rows; i++) {
        for (int j = 0; j < featurePatch.cols; j++) {
            cv::Vec3b pixel = featurePatch.at<cv::Vec3b>(i, j);
            int blue = (int)pixel[0];  // Blue channel
            int green = (int)pixel[1]; // Green channel
            int red = (int)pixel[2];   // Red channel

            // Save the color channels in the format B,G,R
            csvFile << "," << blue << "," << green << "," << red;
        }
    }
    csvFile << "\n";
}

// compute the SSD between two images based on the 7 x 7 pixels extracted
double computeSSD(const std::vector<int> &featureVector1, const std::vector<int> &featureVector2){
    double ssd = 0;
    for (size_t i = 0; i < featureVector1.size(); i++){
        ssd += pow(featureVector1[i] - featureVector2[i], 2);
    };
    return ssd;
}