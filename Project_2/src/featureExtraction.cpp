#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "matchingMethods/baselineMatching.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_directory> <output_csv_file>" << std::endl;
        return -1;
    }

    std::string imageDirectory = argv[1];
    std::string outputCsvFile = argv[2];

    std::ofstream csvFile(outputCsvFile);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening CSV file!" << std::endl;
        return -1;
    }

    // Iterate through all images in the directory and extract features
    for (const auto &entry : std::__fs::filesystem::directory_iterator(imageDirectory)) {
        std::string imagePath = entry.path().string();
        extractAndSaveFeatures(imagePath, csvFile);  // Extract and save features including color info
    }

    csvFile.close();
    std::cout << "Feature extraction with color information completed and saved to " << outputCsvFile << std::endl;

    return 0;
}
