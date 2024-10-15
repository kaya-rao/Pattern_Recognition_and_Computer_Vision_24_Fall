#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream> 
#include <opencv2/opencv.hpp>


// Extract 2D Color Histogram of the image
cv::Mat extractMultiHistogram1(cv::Mat &image){
    int BIN_NUM = 16;
    int binSize = 256 / BIN_NUM;
    cv::Mat histogram = cv::Mat::zeros(BIN_NUM, BIN_NUM, CV_32F);
    
    // loop through each pixel to generate the histogram 
    for (int r = 0; r < (image.rows / 2); r++){
        cv::Vec3b *srcptr = image.ptr<cv::Vec3b>(r);
        for (int c = 0; c < (image.cols / 2); c++){
            int blue = srcptr[c][0];
            int red = srcptr[c][2]; 

            // Locate the bins
            int blueBin = static_cast<int>(blue / 256.0 * binSize);
            int redBin = static_cast<int>(red / 256.0 * binSize);

            // Update histogram
            histogram.at<float>(blueBin, redBin)++;
        }
    }
    // Normalization
    cv::normalize(histogram, histogram, 1, 0, cv::NORM_L1);
    return histogram;
}

// Extract grayscale Histogram of the image
cv::Mat extractMultiHistogram2(cv::Mat &image){
    int BIN_NUM = 256;
    cv::Mat histogram = cv::Mat::zeros(BIN_NUM, 1, CV_32F);  // Proper 1D histogram initialization
    
    // loop through each pixel to generate the histogram 
    for (int r = 0; r < image.rows; r++) {
        for (int c = 0; c < image.cols; c++) {
            int grayValue = image.at<uchar>(r, c);
            histogram.at<float>(grayValue)++;
        }
    }
    // Normalization
    cv::normalize(histogram, histogram, 1, 0, cv::NORM_L1);
    return histogram;
}

// Extract 3D Color Histogram of the image and write into a cvs file
void extractAndSaveBothHistogram(const std::string &imagePath, std::ofstream &csvFile){
    cv::Mat colorImage = cv::imread(imagePath, cv::IMREAD_COLOR);  // Read the image in color
    cv::Mat grayImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);  // Read the image in gray scale
    if (colorImage.empty()) {
        std::cerr << "Error reading color image: " << imagePath << std::endl;
        return;
    }
    if (grayImage.empty()) {
        std::cerr << "Error reading gray image: " << imagePath << std::endl;
        return;
    }

    cv::Mat histogram1 = extractMultiHistogram1(colorImage);
    cv::Mat histogram2 = extractMultiHistogram2(grayImage);

    // Write the image path to the CSV file
    csvFile << imagePath;
    
    // Write the histogram1 to the CSV file
    // Write the histogram to the CSV file
    for (int r = 0; r < histogram1.rows; r++) {
        for (int c = 0; c < histogram1.cols; c++) {
            csvFile << "," << histogram1.at<float>(r, c);
        }
    }
    // Write the histogram1 to the CSV file
    // Iterate through the 1D histogram and save it to CSV after the 3D image

    for (int i = 0; i < histogram2.size[0]; i++){
        csvFile<< "," << histogram2.at<float>(i);
    } 
    csvFile << "\n";
}


double computeHistogramIntersection1D(const cv::Mat &histogram1, const cv::Mat &histogram2){
    double intersection = 0.0;
    // Check that the histograms are 1D and have the same size
    if (histogram1.total() != histogram2.total()) {
        std::cerr << "Histogram dimensions mismatch!" << std::endl;
        return std::numeric_limits<double>::max();
    }
    // Iterate through each bin and compute the intersection
    for (int i = 0; i < histogram1.total(); i++) {
        intersection += std::min(histogram1.at<float>(i), histogram2.at<float>(i));
    }
    return -intersection;  // Negative value for sorting purposes (larger value means more similar)
}
