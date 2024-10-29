/*
  Yunxuan 'Kaya' Rao
  10/22/2024
The collections of the helper functions that's going to apply to the image/live stream
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include "imgProcessing.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <numeric>


// Task 1: grayscale
// Grayscale filter
int greyscale(cv::Mat &src, cv::Mat &dst) {
    // Create an empty image of the same size as src, but with a single channel for grayscale
    dst.create(src.size(), CV_8UC1);
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *srcptr = src.ptr<cv::Vec3b>(i);
        uchar *dstptr = dst.ptr<uchar>(i);
        
        // Apply grayscale transformation
        for (int j = 0; j < src.cols; j++) {
            // Apply the standard grayscale conversion formula
            dstptr[j] = static_cast<uchar>(
                0.299 * srcptr[j][2] +  // Red channel
                0.587 * srcptr[j][1] +  // Green channel
                0.114 * srcptr[j][0]    // Blue channel
            );
        }
    }
    return 0;
}

// Task1: 5 X 5 GaussianBlur blur in greyscale image
int gaussianBlur( cv::Mat &src, cv::Mat &dst ){
    dst.create(src.size(), src.type());

    // Create a temperate image Mat
    cv::Mat tmp = cv::Mat::zeros(src.size(), src.type());
    
    // 1D 5 x 1 blur filter
    int blurFilter[5] = {1, 2, 4, 2, 1};
    int filterSum = 16;
    
    // Horizontal pass
    for (int r = 0; r < src.rows; r++) {
        for (int c = 2; c < src.cols - 2; c++) {
            int newValue = 0;

            for (int j = -2; j <= 2; j++) {
                uchar srcPixel = src.at<uchar>(r, c + j);
                newValue += srcPixel * blurFilter[j + 2];
            }

            tmp.at<uchar>(r, c) = static_cast<uchar>(newValue / filterSum);
        }
    }

    // Vertical pass
    for (int r = 2; r < src.rows - 2; r++) {
        for (int c = 0; c < src.cols; c++) {
            int newValue = 0;

            for (int i = -2; i <= 2; i++) {
                uchar tmpPixel = tmp.at<uchar>(r + i, c);
                newValue += tmpPixel * blurFilter[i + 2];
            }

            dst.at<uchar>(r, c) = static_cast<uchar>(newValue / filterSum);
        }
    }
    return(0);
}


// Task 1: k means thresholding
int kmeansThreshold(const cv::Mat& src, int K) {
    
    // Convert grayscale image to a single row of data
    cv::Mat samples(src.rows * src.cols, 1, CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            samples.at<float>(y + x * src.rows, 0) = static_cast<float>(src.at<uchar>(y, x));
        }
    }

    // K-means clustering
    cv::Mat labels;
    cv::Mat centers;
    kMeans(samples, K, labels, centers, 10);
    
    float mean1 = centers.at<float>(0, 0);
    float mean2 = centers.at<float>(1, 0);

    // Set threshold between two means
    return static_cast<int>((mean1 + mean2) / 2);  
}


// K-means clustering implementation
void kMeans(const cv::Mat& data, int K, cv::Mat& labels, cv::Mat& centers, int maxIterations){
    // Randomly initialize the centroids
    centers = cv::Mat(K, 1, CV_32F);
    std::vector<int> usedIndices;
    for (int i = 0; i < K; ++i) {
        int idx;
        while (std::find(usedIndices.begin(), usedIndices.end(), idx) != usedIndices.end()){
            idx = rand() % data.rows;
        }
        usedIndices.push_back(idx);
        centers.at<float>(i, 0) = data.at<float>(idx, 0);
    }

    labels = cv::Mat(data.rows, 1, CV_32S, cv::Scalar(-1));
    bool converged = false;
    int iterations = 0;

    // Iterate until convergence or max iterations are reached
    while (!converged && iterations < maxIterations) {
        converged = true;

        // Assignment Step
        for (int i = 0; i < data.rows; ++i) {
            float minDistance = std::numeric_limits<float>::max();
            int bestCluster = -1;
            for (int j = 0; j < K; ++j) {
                float distance = std::abs(data.at<float>(i, 0) - centers.at<float>(j, 0));
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = j;
                }
            }

            // If label changes, we haven't converged yet
            if (labels.at<int>(i, 0) != bestCluster) {
                converged = false;
            }
            labels.at<int>(i, 0) = bestCluster;
        }

        // Update Step
        std::vector<float> newCentersSum(K, 0.0f);
        std::vector<int> newCentersCount(K, 0);
        for (int i = 0; i < data.rows; ++i) {
            int cluster = labels.at<int>(i, 0);
            newCentersSum[cluster] += data.at<float>(i, 0);
            newCentersCount[cluster] += 1;
        }

        for (int j = 0; j < K; ++j) {
            if (newCentersCount[j] > 0) {
                centers.at<float>(j, 0) = newCentersSum[j] / newCentersCount[j];
            }
        }

        iterations++;
    }
}

// Task1 - threadshold
int threadshold(cv::Mat &src, cv::Mat &dst, int thresholdValue){
    dst = cv::Mat::zeros(src.size(), CV_8UC1); // empty image same size, single channel
    for(int i=0; i<src.rows; i++) {
        uchar *srcptr = src.ptr<uchar>(i);
        uchar *dstptr = dst.ptr<uchar>(i);
    
        for(int j=0; j < src.cols; j++) {
            if (srcptr[j] < thresholdValue){
                dstptr[j] = 255;
            }
        }
    }
    return(0);
}

// Generate a color palette with three colors: red, green, and yellow
std::vector<cv::Vec3b> generateColorPalette() {
    std::vector<cv::Vec3b> palette;
    palette.push_back(cv::Vec3b(0, 0, 255));   // Red
    palette.push_back(cv::Vec3b(0, 255, 0));   // Green
    palette.push_back(cv::Vec3b(0, 255, 255)); // Yellow
    return palette;
}


// Function to compute and display features for a specified region
FeatureVector computeAndDisplayRegionFeatures(cv::Mat& regionMap, int regionID, cv::Mat& displayImage) {
    // Mask for the current region
    cv::Mat regionMask = (regionMap == regionID);

    // Calculate moments for the region
    cv::Moments moments = cv::moments(regionMask, true);

    // Centroid of the region
    cv::Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

    // Covariance matrix for the region to find the orientation
    double mu20 = moments.mu20 / moments.m00;
    double mu11 = moments.mu11 / moments.m00;
    double mu02 = moments.mu02 / moments.m00;

    // Create covariance matrix
    cv::Mat covarMatrix = (cv::Mat_<double>(2, 2) << mu20, mu11, mu11, mu02);

    // Calculate eigenvalues and eigenvectors
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(covarMatrix, eigenvalues, eigenvectors);

    // Axis orientation (major and minor axes)
    cv::Point2f majorAxis(eigenvectors.at<double>(0, 0), eigenvectors.at<double>(0, 1));
    cv::Point2f minorAxis(eigenvectors.at<double>(1, 0), eigenvectors.at<double>(1, 1));

    // Scale axes for visualization
    float axisLength = 50;  // Adjust as needed for display purposes
    cv::Point2f majorEndPoint = centroid + axisLength * majorAxis;
    cv::Point2f minorEndPoint = centroid + axisLength * minorAxis;

    
    // Draw centroid
    cv::circle(displayImage, centroid, 5, cv::Scalar(0, 0, 255), -1);
  
    // Draw major and minor axes
    cv::line(displayImage, centroid, majorEndPoint, cv::Scalar(255, 0, 0), 2); // Blue for major axis  
    cv::line(displayImage, centroid, minorEndPoint, cv::Scalar(0, 255, 0), 2); // Green for minor axis
   
    // Find contours for the region
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(regionMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double huMomentsArray[7];
    float percentFilled;
    float heightWidthRatio;


    // Check if we have any valid contours
    if (!contours.empty()) {
        // Use the largest contour for the bounding box (assuming only one region is present in the mask)
        int largestContourIndex = 0;  // In case there are multiple contours, you may need to choose which one to use
        cv::RotatedRect orientedBox = cv::minAreaRect(contours[largestContourIndex]);

        // Extract the points of the oriented bounding box
        cv::Point2f boxPoints[4];
        orientedBox.points(boxPoints);

        // Draw the oriented bounding box
        for (int j = 0; j < 4; j++) {
            cv::line(displayImage, boxPoints[j], boxPoints[(j + 1) % 4], cv::Scalar(0, 255, 255), 2); // Draw yellow bounding box
        }

        // Calculate and display percent filled and bounding box height/width ratio
        // Calculate bounding box area
        float boundingBoxArea = orientedBox.size.width * orientedBox.size.height;

        // Calculate region area using contour area
        double regionArea = cv::contourArea(contours[largestContourIndex]);

        // Calculate percent filled (region area / bounding box area * 100)
        percentFilled = (boundingBoxArea > 0) ? (regionArea / boundingBoxArea * 100.0) : 0;

        // Calculate height/width ratio
        // Always use the longer length / shorter length to record consist results
        // Since object can be rotating
        heightWidthRatio = 0;
        if (orientedBox.size.height > orientedBox.size.width && orientedBox.size.width > 0){
            heightWidthRatio = orientedBox.size.height / orientedBox.size.width;
        } else if (orientedBox.size.width > orientedBox.size.height && orientedBox.size.height > 0){
            heightWidthRatio = orientedBox.size.width / orientedBox.size.height;
        }
        // Display percent filled and height/width ratio on the image
        std::ostringstream textStream;
        textStream << "% Filled: " << std::fixed << std::setprecision(2) << percentFilled << "%";
        cv::putText(displayImage, textStream.str(), boxPoints[1], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        
        textStream.str(""); // Clear the stream
        textStream << "H/W Ratio: " << std::fixed << std::setprecision(2) << heightWidthRatio;
        cv::putText(displayImage, textStream.str(), boxPoints[2], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        // Calculate and display Hu moments
        cv::Moments huMomentsCalc = cv::moments(contours[largestContourIndex]);
        cv::HuMoments(huMomentsCalc, huMomentsArray);
        // Display Hu moments on the image
        for (int k = 0; k < 7; k++) {
            textStream.str(""); // Clear the stream
            textStream << "Hu Moment " << (k + 1) << ": " << std::fixed << std::setprecision(2) << huMomentsArray[k];
            cv::putText(displayImage, textStream.str(), cv::Point(10, 80 + k * 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        }
    }
    // Convert array to vector
    std::vector<double> huMoments(huMomentsArray, huMomentsArray + 7);  
    return {centroid, majorAxis, minorAxis, percentFilled, heightWidthRatio, huMoments};
}


// Calculate the Euclidean distance between two points
double euclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

// Ask user for a label and save feature vector to file
void saveFeatureVector(FeatureVector featureVector, const std::string& label, const std::string& filename) {

    std::ofstream csvFile(filename, std::ios::app);
    if (!csvFile.is_open()) {
        std::cerr << "Error opening CSV file!" << std::endl;
        return;
    }

    if (csvFile.is_open()) {
        // write label to the file
        csvFile << label;

        // write other features
        csvFile << "," << featureVector.centroid.x << "," << featureVector.centroid.y;
        csvFile << "," << featureVector.majorAxis.x << "," << featureVector.majorAxis.y;
        csvFile << "," << featureVector.minorAxis.x << "," << featureVector.minorAxis.y;
        csvFile << "," << featureVector.percentFilled;
        csvFile << "," << featureVector.heightWidthRatio;

        // write hu moment
        for (const double& huMoment : featureVector.huMoments) {
            csvFile << "," << huMoment;
        }

        csvFile << "\n";
        csvFile.close();
        std::cout << "Feature vector saved for label: " << label << std::endl;
    } else {
        std::cerr << "Error opening file for writing." << std::endl;
    }
}


// Load database from csv file
std::vector<std::pair<std::string, FeatureVector>> loadDatabase(const std::string& filename) {
    std::vector<std::pair<std::string, FeatureVector>> database;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return database;
    }

    std::string line;
    //std::getline(file, line); // No header row

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        FeatureVector fv;
        std::string label, token;

        // Read label
        std::getline(ss, label, ',');
        //std::cout<<label<<std::endl;

        // Read centroid, majorAxis, minorAxis, percentFilled, and heightWidthRatio
        std::getline(ss, token, ','); fv.centroid.x = std::stod(token);
        std::getline(ss, token, ','); fv.centroid.y = std::stod(token);
        std::getline(ss, token, ','); fv.majorAxis.x = std::stod(token);
        std::getline(ss, token, ','); fv.majorAxis.y = std::stod(token);
        std::getline(ss, token, ','); fv.minorAxis.x = std::stod(token);
        std::getline(ss, token, ','); fv.minorAxis.y = std::stod(token);
        std::getline(ss, token, ','); fv.percentFilled = std::stod(token);
        std::getline(ss, token, ','); fv.heightWidthRatio = std::stod(token);

        // Read Hu moments
        fv.huMoments.resize(7);
        for (int i = 0; i < 7; ++i) {
            if (std::getline(ss, token, ',')) {
                fv.huMoments[i] = std::stod(token);
                //std::cout<<token<<std::endl;
            } else {
                std::cerr << "Error: Missing Hu Moment data in line: " << line << std::endl;
                fv.huMoments[i] = 0.0;
            }
        }

        // Add the label and feature vector as a pair to the database
        database.emplace_back(label, fv);
    }
    file.close();
    return database;
}


// Calculate standard deviations for each feature (percentFilled, heightWidthRatio, and huMoments)
FeatureVector computeStandardDeviations(const std::vector<std::pair<std::string, FeatureVector>>& database) {
    FeatureVector stdevs;
    size_t numEntries = database.size();

    // Initialize sums for each feature
    double percentFilledSum = 0.0;
    double heightWidthRatioSum = 0.0;
    std::vector<double> huMomentsSum(7, 0.0);

    // Step 1: Calculate the mean for each feature
    for (const auto& entry : database) {
        const FeatureVector& fv = entry.second;
        percentFilledSum += fv.percentFilled;
        heightWidthRatioSum += fv.heightWidthRatio;
        
        for (size_t i = 0; i < fv.huMoments.size(); ++i) {
            huMomentsSum[i] += fv.huMoments[i];
        }
    }

    // Calculate mean for each feature
    FeatureVector means;
    means.percentFilled = percentFilledSum / numEntries;
    means.heightWidthRatio = heightWidthRatioSum / numEntries;
    means.huMoments.resize(7);
    for (size_t i = 0; i < huMomentsSum.size(); ++i) {
        means.huMoments[i] = huMomentsSum[i] / numEntries;
    }

    // Calculate the variance 
    double percentFilledDiff = 0.0;
    double heightWidthRatioDiff = 0.0;
    std::vector<double> huMomentsDiff(7, 0.0);

    for (const auto& entry : database) {
        const FeatureVector& fv = entry.second;
        percentFilledDiff += std::pow(fv.percentFilled - means.percentFilled, 2);
        heightWidthRatioDiff += std::pow(fv.heightWidthRatio - means.heightWidthRatio, 2);
        
        for (size_t i = 0; i < fv.huMoments.size(); ++i) {
            huMomentsDiff[i] += std::pow(fv.huMoments[i] - means.huMoments[i], 2);
        }
    }

    // Compute standard deviations 
    stdevs.percentFilled = std::sqrt(percentFilledDiff / numEntries);
    stdevs.heightWidthRatio = std::sqrt(heightWidthRatioDiff / numEntries);
    stdevs.huMoments.resize(7);
    for (size_t i = 0; i < huMomentsDiff.size(); ++i) {
        stdevs.huMoments[i] = std::sqrt(huMomentsDiff[i] / numEntries);
    }

    // Since centroid, majorAxis, and minorAxis are not rotation invariant
    // I'm not going to use them for matching, but it's easier to use a pre exist dataStructure
    stdevs.centroid = {0, 0};
    stdevs.majorAxis = {0, 0};
    stdevs.minorAxis = {0, 0};

    return stdevs;
}

// Calculate Scaled Euclidean
// only use percentFilled, heightWidthRatio, and huMoments
double scaledEuclideanDistance(const FeatureVector& fv1, const FeatureVector& fv2, const FeatureVector& stdevs) {
    double distance = 0.0;

    // Giving weight to the features
    double weightPercentFilled = 3.0;
    double weightHeightWidthRatio = 3.0;
    // The last hu moment is not rotation invariant
    std::vector<double> huMomentsWeights = {1.5, 1.5, 1.0, 1.0, 0.5, 0.5, 0};

    // Scaled distance for percentFilled
    if (stdevs.percentFilled != 0) {
        distance += std::pow((fv1.percentFilled - fv2.percentFilled) / stdevs.percentFilled, 2);
    }

    // Scaled distance for heightWidthRatio
    if (stdevs.heightWidthRatio != 0) {
        distance += std::pow((fv1.heightWidthRatio - fv2.heightWidthRatio) / stdevs.heightWidthRatio, 2);
    }

    // Scaled distance for Hu moments
    for (size_t i = 0; i < fv1.huMoments.size(); i++) {
        if (stdevs.huMoments[i] != 0) {  // Only include in distance if stdev is non-zero
            distance += std::pow((fv1.huMoments[i] - fv2.huMoments[i]) / stdevs.huMoments[i], 2);
        }
    }

    return std::sqrt(distance);  // Return the Euclidean distance
}

// Find the nearest neighbor in the database
std::string classifyObject(const FeatureVector& newFV, const std::vector<std::pair<std::string, FeatureVector>>& database, const FeatureVector& stdevs) {
    double minDistance = std::numeric_limits<double>::max();
    std::string nearestLabel;

    for (const auto& entry : database) {
        const auto& fv = entry.second;
        double distance = scaledEuclideanDistance(newFV, fv, stdevs);
        //std::cout << "Comparing with label: " << entry.first << ", Distance: " << distance << std::endl;
        if (distance < minDistance) {
            minDistance = distance;
            nearestLabel = entry.first;
        }
    }
    //std::cout << "Nearest label: " << nearestLabel << ", Min Distance: " << minDistance << std::endl;
    return nearestLabel;
}


// update Confusion Matrix
void updateConfusionMatrix(const std::string& trueLabel, const std::string& predictedLabel, std::vector<std::vector<int>>& confusionMatrix, const std::map<std::string, int>& labelIndex) {
    auto trueIt = labelIndex.find(trueLabel);
    auto predIt = labelIndex.find(predictedLabel);
    if (trueIt != labelIndex.end() && predIt != labelIndex.end()) {
        int trueIndex = trueIt->second;
        int predictedIndex = predIt->second;
        confusionMatrix[trueIndex][predictedIndex]++;
    } 
}


// Print confusion matrixstd::vector<std::vector<int>> confusionMatrix
void printConfusionMatrix(const std::vector<std::vector<int>>& confusionMatrix, const std::map<std::string, int>& labelIndex) {
    std::cout << "Confusion Matrix:" << std::endl;

    // Title
    std::cout << "     ";
    for (const auto& label : labelIndex) {
        std::cout << label.first << " ";
    }
    std::cout << std::endl;

    // content
    for (size_t i = 0; i < confusionMatrix.size(); ++i) {
        std::cout << std::left << std::setw(6) << std::next(labelIndex.begin(), i)->first << " ";
        for (size_t j = 0; j < confusionMatrix[i].size(); ++j) {
            std::cout << std::setw(6) << confusionMatrix[i][j];
        }
        std::cout << std::endl;
    }
}

// Calculate Cosine Distance
double cosineDistance(const FeatureVector& fv1, const FeatureVector& fv2) {

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    // percentFilled & heightWidthRatio 
    dotProduct += fv1.percentFilled * fv2.percentFilled + fv1.heightWidthRatio * fv2.heightWidthRatio;
    normA += std::pow(fv1.percentFilled, 2) + std::pow(fv1.heightWidthRatio, 2);
    normB += std::pow(fv2.percentFilled, 2) + std::pow(fv2.heightWidthRatio, 2);

    // Hu Moments: use the first six values
    for (size_t i = 0; i < 6; i++) {
        dotProduct += fv1.huMoments[i] * fv2.huMoments[i];
        normA += std::pow(fv1.huMoments[i], 2);
        normB += std::pow(fv2.huMoments[i], 2);
    }

    double cosineSimilarity = dotProduct / (std::sqrt(normA) * std::sqrt(normB));
    return 1.0 - cosineSimilarity; 
}

std::string classifyObjectUseCosin(const FeatureVector& newFV, const std::vector<std::pair<std::string, FeatureVector>>& database) {
    double minDistance = std::numeric_limits<double>::max();
    std::string nearestLabel;

    for (const auto& entry : database) {
        const auto& fv = entry.second;
        double distance = cosineDistance(newFV, fv);

        if (distance < minDistance) {
            minDistance = distance;
            nearestLabel = entry.first;
        }
    }

    return nearestLabel;
}