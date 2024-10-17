#include "deepNetworkMatching.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream> 
#include <filesystem>


// compute the SSD between two images features
double computeDnnSSD(const std::vector<float> &featureVector1, const std::vector<float> &featureVector2){
    double ssd = 0;
    for (size_t i = 0; i < featureVector1.size(); i++){
        ssd += pow(featureVector1[i] - featureVector2[i], 2);
    };
    return ssd;
}

// Extract the filename from a path, 
// use to clean and match the image path for dnn features
std::string extractFilename(const std::string& path) {
    // Use std::filesystem to extract the filename
    std::__fs::filesystem::path filePath(path);
    return filePath.filename().string();
}