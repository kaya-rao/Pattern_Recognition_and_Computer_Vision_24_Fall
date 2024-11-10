/*
  Yunxuan 'Kaya' Rao
  11/03/2024
  Display live stream video and apply filters on it.
 */
#include <opencv2/opencv.hpp>
#include <iostream>

// -------------------------- Task 7: Detect Robust Features using Harris Corner -------------------------- //
int main() {
    // Open the default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int threshold = 160;

    while (true) {

        // Start a video stream
        cv::Mat frame, gray, dst;
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame." << std::endl;
            break;
        }

        // Convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Harris corner detection
        dst = cv::Mat::zeros(gray.size(), CV_32FC1);
        cv::cornerHarris(gray, dst, blockSize, apertureSize, k);

        // Normalize and threshold
        cv::Mat dst_norm, dst_norm_scaled;
        cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
        cv::convertScaleAbs(dst_norm, dst_norm_scaled);

        // Draw circles on detected corners
        for (int i = 0; i < dst_norm.rows; i++) {
            for (int j = 0; j < dst_norm.cols; j++) {
                if ((int)dst_norm.at<float>(i, j) > threshold) {
                    cv::circle(frame, cv::Point(j, i), 5, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                }
            }
        }

        // Display result
        cv::imshow("Harris Corners", frame);

        // Break on 'q' key press
        if (cv::waitKey(30) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
