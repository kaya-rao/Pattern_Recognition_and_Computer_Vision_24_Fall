#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>  // For handling CSV parsing
#include "matchingMethods/baselineMatching.h"
#include "matchingMethods/histogramMatching.h"
#include "matchingMethods/multiHistogramMatching.h"
#include "matchingMethods/textureColor.h"
#include "matchingMethods/deepNetworkMatching.h"




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
    cv::Mat grayTargetImage = cv::imread(targetImagePath, cv::IMREAD_GRAYSCALE);
    if (targetImage.empty()) {
        std::cerr << "Error reading target image!" << std::endl;
        return -1;
    }
    
    std::vector<int> targetFeatureVector;
    cv::Mat targetHistogram;
    cv::Mat targetMultiHistogram1;
    cv::Mat targetMultiHistogram2;
    RGBHistogram targetRGBHistogram;
    std::vector<float> targetTextureHistogram;
    std::vector<float> targetDnnVector;

    if(distanceMatrix == "baseline"){
        targetFeatureVector = extractSSDFeatureVector(targetImage);
    } else if (distanceMatrix == "histogram"){
        targetHistogram = extract2DColorHistogram(targetImage);
    } else if (distanceMatrix == "multihist"){
        targetMultiHistogram1 = extractMultiHistogram1(targetImage);
        targetMultiHistogram2 = extractMultiHistogram2(grayTargetImage);    
    } else if (distanceMatrix == "tc"){
        targetRGBHistogram = extractWholeColorHistogram(targetImage);
        targetTextureHistogram = extractTextureHistogram(targetImage);
    } else if (distanceMatrix == "dnn"){
        // Read feature vectors from CSV and compute SSD
        std::ifstream csvFile(featureCsvFile);
        std::string line;
        std::string extractedImagePath = extractFilename(targetImagePath);
        // Save all the features
        while (std::getline(csvFile, line)) {
            std::istringstream stream(line);
            std::string imagePath;
            // Read the image path from the CSV line
            std::getline(stream, imagePath, ',');
            if (extractedImagePath == imagePath) {
                std::string value;
                while (std::getline(stream, value, ',')) {
                    targetDnnVector.push_back(std::stoi(value));  // Convert string to int
                }
                break;
            }
        }
        csvFile.close();
    } else {
        std::cerr << "Error finding distance Matrix!" << std::endl;
        return -1;
    }
    

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
        }  else if (distanceMatrix == "multihist"){
            distance = 0;
            // convert csv to histogram
            // Create an empty histogram with the expected size
            int BIN_NUM = 16;
            int histSize[] = {BIN_NUM, BIN_NUM, BIN_NUM};
            cv::Mat histogram1 = cv::Mat::zeros(3, histSize, CV_32F);  // Proper 3D histogram initialization

            std::string val;
            int idx = 0;
            // Read each bin value
            while (idx < 256 && std::getline(stream, val, ',')) {
                int r = idx / BIN_NUM;
                int c = idx % BIN_NUM;
                histogram1.at<float>(r, c) = std::stof(val);
                idx++;
            }
            // Convert CSV to histogram2 (1D)
            int GRAY_BIN_NUM = 256;
            cv::Mat histogram2 = cv::Mat::zeros(1, GRAY_BIN_NUM, CV_32F);  // Proper 1D grayscale histogram initialization

            for (int i = 0; i < GRAY_BIN_NUM; i++) {
                if (std::getline(stream, val, ',')) {
                    histogram2.at<float>(0, i) = std::stof(val);
                } else {
                    std::cerr << "Error reading value for grayscale histogram2!" << std::endl;
                }
            }
            // Compute histogram intersections
            distance += computeHistogramIntersection(targetMultiHistogram1, histogram1);
            distance += computeHistogramIntersection1D(targetMultiHistogram2, histogram2);
            distance /= 2;

        }  else if (distanceMatrix == "tc"){
            // ------------- RGB Histogram ------------- // 
            // Define the size of the bins
            int binNum1 = 8;
            int binSize = 256 / binNum1;
            RGBHistogram rbgHistogram;
            rbgHistogram.blueHist.resize(binNum1);
            rbgHistogram.greenHist.resize(binNum1);
            rbgHistogram.redHist.resize(binNum1);

            // Extract from csv 
            int idx = 0;
            // Read each bin value
            std::string value;
            while (idx < 24) {
                std::getline(stream, value, ',');
                rbgHistogram.blueHist[static_cast<int>(idx / 3)] = std::stof(value);
                idx += 1;

                std::getline(stream, value, ',');
                rbgHistogram.greenHist[static_cast<int>(idx / 3)] = std::stof(value);
                idx += 1;

                std::getline(stream, value, ',');
                rbgHistogram.redHist[static_cast<int>(idx / 3)] = std::stof(value);
                idx += 1;
            }
            double rbgHistogramDistance = computeHistogramIntersection3D(targetRGBHistogram, rbgHistogram);

            // ------------- Texture Histogram ------------- // 
            std::vector<float> textureHistogram(16, 0.0f);
            int binNum2 = 16;

            // Extract from csv 
            // Read each bin value
            int textureIdx = 0;
            while (textureIdx < binNum2) {
                std::getline(stream, value, ',');
                textureHistogram[textureIdx] = std::stof(value);
                textureIdx += 1;
            }
            double singleChannelIntersection = - computeSingleChannelIntersection(targetTextureHistogram ,textureHistogram);
            // Equal weighted
            distance = (rbgHistogramDistance + singleChannelIntersection) / 2;
        } else if (distanceMatrix == "dnn"){
            std::vector<float> dnnVector;
            // Read the feature vector values
            std::string value;
            while (std::getline(stream, value, ',')) {
                dnnVector.push_back(std::stoi(value));
            }
            distance = computeDnnSSD(targetDnnVector, dnnVector);
        } else {
            std::cerr << "Error finding distance Matrix!" << std::endl;
            return -1;
        }
        
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
