#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>  // For handling CSV parsing
#include "matchingMethods/baselineMatching.h"

// Struct to store image feature and distance
struct ImageFeature {
    std::string imagePath;
    std::vector<int> featureVector;
    double distance;
};

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <target_image> <feature_csv_file> <N>" << std::endl;
        return -1;
    }

    std::string targetImagePath = argv[1];
    std::string featureCsvFile = argv[2];
    int N = std::stoi(argv[3]);

    // Read target image and compute its feature vector
    cv::Mat targetImage = cv::imread(targetImagePath, cv::IMREAD_COLOR);
    if (targetImage.empty()) {
        std::cerr << "Error reading target image!" << std::endl;
        return -1;
    }

    std::vector<int> targetFeatureVector = extractFeatureVector(targetImage);

    // Read feature vectors from CSV and compute SSD
    std::ifstream csvFile(featureCsvFile);
    std::string line;
    std::vector<ImageFeature> imageFeatures;

    while (std::getline(csvFile, line)) {
        std::istringstream stream(line);
        std::string imagePath;
        std::vector<int> featureVector;
        
        // First read the image path from the CSV line
        std::getline(stream, imagePath, ',');

        // Then read the feature vector values
        std::string value;
        while (std::getline(stream, value, ',')) {
            featureVector.push_back(std::stoi(value));
        }

        // Compute SSD for each image
        double ssd = computeSSD(targetFeatureVector, featureVector);
        imageFeatures.push_back({imagePath, featureVector, ssd});
    }

    csvFile.close();

    // Sort images by SSD
    std::sort(imageFeatures.begin(), imageFeatures.end(), [](const ImageFeature &a, const ImageFeature &b) {
        return a.distance < b.distance;
    });

    // Print top N matches
    std::cout << "Top " << N << " matches:" << std::endl;
    for (int i = 0; i < N && i < imageFeatures.size(); i++) {
        std::cout << imageFeatures[i].imagePath << " with SSD: " << imageFeatures[i].distance << std::endl;
    }

    return 0;
}
