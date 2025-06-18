#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>

int main() {
    std::cout << "Pokrecem obradu slike...\n";

    cv::Mat img = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Nije moguce ucitati sliku!\n";
        return -1;
    }

    cv::Mat grad(img.rows, img.cols, CV_8UC1);

    double t0 = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < img.rows - 1; y++) {
        for (int x = 1; x < img.cols - 1; x++) {
            int gx = -img.at<uchar>(y-1, x-1) - 2*img.at<uchar>(y, x-1) - img.at<uchar>(y+1, x-1)
                     + img.at<uchar>(y-1, x+1) + 2*img.at<uchar>(y, x+1) + img.at<uchar>(y+1, x+1);
            int gy = -img.at<uchar>(y-1, x-1) - 2*img.at<uchar>(y-1, x) - img.at<uchar>(y-1, x+1)
                     + img.at<uchar>(y+1, x-1) + 2*img.at<uchar>(y+1, x) + img.at<uchar>(y+1, x+1);
            grad.at<uchar>(y, x) = std::min(255, (int)sqrt(gx*gx + gy*gy));
        }
    }

    double t1 = omp_get_wtime();
    std::cout << "Vrijeme CPU obrade (OpenMP): " << (t1 - t0) << " sekundi\n";

    cv::imwrite("output_cpu.jpg", grad);

    cv::imshow("Rezultat Sobel filtera", grad);
    cv::waitKey(0);

    return 0;
}