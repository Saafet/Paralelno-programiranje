#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <cmath>

int main() {
    std::cout << "Pokrecem obradu slike...\n";

    cv::Mat img = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Nije moguce ucitati sliku!\n";
        return -1;
    }

    cv::Mat grad(img.rows, img.cols, CV_8UC1);

    const uchar* src = img.ptr<uchar>();
    uchar* dst = grad.ptr<uchar>();
    size_t step = img.step;

    double t0 = omp_get_wtime();

    #pragma omp parallel for
    for (int y = 1; y < img.rows - 1; y++) {
        for (int x = 1; x < img.cols - 1; x++) {
            int idx = y * step + x;

            int gx = -src[idx - step - 1] - 2 * src[idx - 1] - src[idx + step - 1]
                     + src[idx - step + 1] + 2 * src[idx + 1] + src[idx + step + 1];

            int gy = -src[idx - step - 1] - 2 * src[idx - step] - src[idx - step + 1]
                     + src[idx + step - 1] + 2 * src[idx + step] + src[idx + step + 1];

            
            int val = abs(gx) + abs(gy);
            dst[idx] = (uchar)(val > 255 ? 255 : val);
        }
    }

    double t1 = omp_get_wtime();

    std::cout << "Vrijeme CPU obrade: " << (t1 - t0) * 1000 << " ms\n";

    cv::imwrite("output_cpu.jpg", grad);

    cv::imshow("Rezultat Sobel filtera", grad);
    cv::waitKey(0);

    return 0;
}
