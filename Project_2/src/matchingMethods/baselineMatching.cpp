#include "baselineMatching.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream> 


// Extract the 7 x 7 pixels from the center and write into a cvs file
// Keep the color channels instead of converting to grayscale

std::vector<int> extractSSDFeatureVector(cv::Mat &image) {
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
void extractAndSaveSSDFeatures(const std::string &imagePath, std::ofstream &csvFile) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);  // Read the image in color
    if (image.empty()) {
        std::cerr << "Error reading image: " << imagePath << std::endl;
        return;
    }

    // Get the feature vector by reusing extractFeatureVector
    std::vector<int> featureVector = extractSSDFeatureVector(image);

    // Write the image path to the CSV file
    csvFile << imagePath;
    
    // Write the feature vector to the CSV file
    for (const auto &value : featureVector) {
        csvFile << "," << value;
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