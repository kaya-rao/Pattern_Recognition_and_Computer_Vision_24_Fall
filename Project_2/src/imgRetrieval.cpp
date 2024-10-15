#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>  // For handling CSV parsing
#include "matchingMethods/baselineMatching.h"
#include "matchingMethods/histogramMatching.h"

// Struct to store image feature and distance
struct ImageFeature {
    std::string imagePath;
    double distance;
};

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <target_image> <feature_csv_file> <N> <distance_matrix>" << std::endl;
        return -1;
    }

    std::string targetImagePath = argv[1];
    std::string featureCsvFile = argv[2];
    int N = std::stoi(argv[3]);
    std::string distanceMatrix = argv[4];


    // Read target image and compute its feature vector
    cv::Mat targetImage = cv::imread(targetImagePath, cv::IMREAD_COLOR);
    if (targetImage.empty()) {
        std::cerr << "Error reading target image!" << std::endl;
        return -1;
    }
    std::vector<int> targetFeatureVector;
    cv::Mat targetHistogram;
    if(distanceMatrix == "baseline"){
        targetFeatureVector = extractSSDFeatureVector(targetImage);
    } else if (distanceMatrix == "histogram"){
        targetHistogram = extract2DColorHistogram(targetImage);
    } /*else if (distanceMatrix == "tc"){
        std::vector<int> targetFeatureVector = extractSSDFeatureVector(targetImage);
    } else if (distanceMatrix == "dnn"){
        std::vector<int> targetFeatureVector = extractSSDFeatureVector(targetImage);
    } else {
        std::cerr << "Error finding distance Matrix!" << std::endl;
        return -1;
    }*/
    

    // Read feature vectors from CSV and compute SSD
    std::ifstream csvFile(featureCsvFile);
    std::string line;
    std::vector<ImageFeature> imageFeatures;

    while (std::getline(csvFile, line)) {
        std::istringstream stream(line);
        std::string imagePath;
        // Read the image path from the CSV line
        std::getline(stream, imagePath, ',');
        double distance;

        // Compute distance for each image
        if(distanceMatrix == "baseline"){
            std::vector<int> featureVector;
            // Read the feature vector values
            std::string value;
            while (std::getline(stream, value, ',')) {
                featureVector.push_back(std::stoi(value));
            }
            distance = computeSSD(targetFeatureVector, featureVector);
        } else if (distanceMatrix == "histogram"){
            // convert csv to histogram
            // Create an empty histogram with the expected size
            int BIN_NUM = 16;
            cv::Mat histogram = cv::Mat::zeros(BIN_NUM, BIN_NUM, CV_32F);
            int idx = 0;

            // Read each bin value
            std::string value;
            while (std::getline(stream, value, ',')) {
                int r = idx / BIN_NUM;
                int c = idx % BIN_NUM;
                histogram.at<float>(r, c) = std::stof(value);
                idx++;
            }
            distance = computeHistogramIntersection(targetHistogram, histogram);
        } /* else if (distanceMatrix == "tc"){
            std::vector<int> targetFeatureVector = extractSSDFeatureVector(targetImage);
        } else if (distanceMatrix == "dnn"){
            std::vector<int> targetFeatureVector = extractSSDFeatureVector(targetImage);
        } else {
            std::cerr << "Error finding distance Matrix!" << std::endl;
            return -1;
        }*/
        
        imageFeatures.push_back({imagePath, distance});
    }

    csvFile.close();

    // Sort images by distance
    std::sort(imageFeatures.begin(), imageFeatures.end(), [](const ImageFeature &a, const ImageFeature &b) {
        return a.distance < b.distance;
    });

    // Print top N matches
    std::cout << "Top " << N << " matches:" << std::endl;
    for (int i = 0; i < N && i < imageFeatures.size(); i++) {
        std::cout << imageFeatures[i].imagePath << " with distance: " << imageFeatures[i].distance << std::endl;
    }

    return 0;
}
