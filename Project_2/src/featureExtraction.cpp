#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "matchingMethods/baselineMatching.h"
#include "matchingMethods/histogramMatching.h"
#include "matchingMethods/multiHistogramMatching.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <output_csv_file> <distance_matrix>" << std::endl;
        return -1;
    }

    std::string imageDirectory = argv[1];
    std::string outputCsvFile = argv[2];
    std::string distanceMatrix = argv[3];

    std::ofstream csvFile(outputCsvFile);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening CSV file!" << std::endl;
        return -1;
    }

    // Iterate through all images in the directory and extract features
    for (const auto &entry : std::__fs::filesystem::directory_iterator(imageDirectory)) {
        std::string imagePath = entry.path().string();
        
        // Extract and save features according to the distance matrix of choice 
        if(distanceMatrix == "baseline"){
            extractAndSaveSSDFeatures(imagePath, csvFile);
        } else if (distanceMatrix == "histogram"){
            extractAndSaveHistogram(imagePath, csvFile);
        } else if (distanceMatrix == "multihist"){
            extractAndSaveBothHistogram(imagePath, csvFile);
        } else {
            std::cerr << "Error: Unsupported distance matrix '" << distanceMatrix << "'!" << std::endl;
            csvFile.close();
            return -1;  // Exit if invalid matrix is provided
        }
    }

    csvFile.close();
    std::cout << "Feature extraction with color information completed and saved to " << outputCsvFile << std::endl;

    return 0;
}
