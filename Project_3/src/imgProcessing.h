/*
  Yunxuan 'Kaya' Rao
  10/22/2024
The header of the helper functions that's going to apply to the image/live stream
 */
#ifndef IMG_PROCESSING
#define IMG_PROCESSING

#include <opencv2/opencv.hpp>


// Structure to hold the features
struct FeatureVector {
    cv::Point2d centroid;
    cv::Point2d majorAxis;
    cv::Point2d minorAxis;
    double percentFilled;
    double heightWidthRatio;
    std::vector<double> huMoments;
};

// Task 1: grayscale
// Grayscale filter
int greyscale(cv::Mat &src, cv::Mat &dst);

// Task1: 5 X 5 GaussianBlur blur
int gaussianBlur( cv::Mat &src, cv::Mat &dst);

// Task 1: k means thresholding
int kmeansThreshold(const cv::Mat& input, int K = 2);

void kMeans(const cv::Mat& data, int K, cv::Mat& labels, cv::Mat& centers, int maxIterations = 100);

int threadshold(cv::Mat &src, cv::Mat &dst, int thresholdValue);

// Generate a color palette with three colors: red, green, and yellow
std::vector<cv::Vec3b> generateColorPalette();

FeatureVector computeAndDisplayRegionFeatures(cv::Mat& regionMap, int regionID, cv::Mat& displayImage);

double euclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2);

void saveFeatureVector(FeatureVector, const std::string& label, const std::string& filename = "features.csv");

std::vector<std::pair<std::string, FeatureVector>> loadDatabase(const std::string& filename);

FeatureVector computeStandardDeviations(const std::vector<std::pair<std::string, FeatureVector>>& database);

double scaledEuclideanDistance(const FeatureVector& fv1, const FeatureVector& fv2, const FeatureVector& stdevs);


std::string classifyObject(const FeatureVector& newFV, const std::vector<std::pair<std::string, FeatureVector>>& database, const FeatureVector& stdevs);

void updateConfusionMatrix(const std::string& trueLabel, const std::string& predictedLabel, std::vector<std::vector<int>>& confusionMatrix, const std::map<std::string, int>& labelIndex);

void printConfusionMatrix(const std::vector<std::vector<int>>& confusionMatrix, const std::map<std::string, int>& labelIndex);

double cosineDistance(const FeatureVector& fv1, const FeatureVector& fv2);

std::string classifyObjectUseCosin(const FeatureVector& newFV, const std::vector<std::pair<std::string, FeatureVector>>& database);

#endif // Processings
