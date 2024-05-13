#include <iostream>
#include <vector>
#include <chrono>
#include <complex>
#include <fftw3.h>
#include <Accelerate/Accelerate.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <random>

void print_performance(const std::string &label, int size, const std::chrono::duration<double> &duration) {
    std::cout << label << " for size " << size << ": " << duration.count() << " seconds" << std::endl;
}

void basic_fft(const std::vector<double>& input, std::vector<std::complex<double>>& output) {
    int N = input.size();
    fftw_complex *in, *out;
    fftw_plan p;

    in = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
    out = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));

    for (int i = 0; i < N; ++i) {
        in[i][0] = input[i];
        in[i][1] = 0.0;
    }

    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    for (int i = 0; i < N; ++i) {
        output[i] = std::complex<double>(out[i][0], out[i][1]);
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
}

void accelerate_fft(const std::vector<double>& input, std::vector<std::complex<double>>& output) {
    int N = input.size();
    DSPDoubleSplitComplex temp;
    temp.realp = (double *)malloc(sizeof(double) * N/2);
    temp.imagp = (double *)malloc(sizeof(double) * N/2);

    vDSP_ctozD((const DSPDoubleComplex *)input.data(), 2, &temp, 1, N/2);
    vDSP_fft_zipD(vDSP_create_fftsetupD(log2(N), FFT_RADIX2), &temp, 1, log2(N), FFT_FORWARD);

    for (int i = 0; i < N/2; ++i) {
        output[i] = std::complex<double>(temp.realp[i], temp.imagp[i]);
    }

    free(temp.realp);
    free(temp.imagp);
}

void basic_convolution(const cv::Mat& input, cv::Mat& output) {
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 1,  1, 1,
                                              1, -8, 1,
                                              1,  1, 1);
    cv::filter2D(input, output, -1, kernel);
}

void accelerate_convolution(const cv::Mat& input, cv::Mat& output) {
    vImage_Buffer src = { input.data, static_cast<vImagePixelCount>(input.rows), static_cast<vImagePixelCount>(input.cols), static_cast<size_t>(input.step) };
    vImage_Buffer dest = { output.data, static_cast<vImagePixelCount>(output.rows), static_cast<vImagePixelCount>(output.cols), static_cast<size_t>(output.step) };

    int16_t kernel[9] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
    vImage_Error error = vImageConvolve_Planar8(&src, &dest, NULL, 0, 0, kernel, 3, 3, 1, NULL, kvImageEdgeExtend);

    if (error != kvImageNoError) {
        std::cerr << "Error in vImage convolution: " << error << std::endl;
    }
}

int main() {
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096}; // Sizes to test

    for (int N : sizes) {
        std::vector<double> fft_input(N, 1.0);
        std::vector<std::complex<double>> fft_output(N);
        auto start = std::chrono::high_resolution_clock::now();
        basic_fft(fft_input, fft_output);
        auto end = std::chrono::high_resolution_clock::now();
        print_performance("Basic FFT", N, end - start);

        start = std::chrono::high_resolution_clock::now();
        accelerate_fft(fft_input, fft_output);
        end = std::chrono::high_resolution_clock::now();
        print_performance("Accelerate FFT", N, end - start);

        cv::Mat image = cv::Mat::ones(N, N, CV_8UC1);
        cv::randu(image, cv::Scalar(0), cv::Scalar(255));
        cv::Mat processed_image = cv::Mat::zeros(N, N, CV_8UC1);

        start = std::chrono::high_resolution_clock::now();
        basic_convolution(image, processed_image);
        end = std::chrono::high_resolution_clock::now();
        print_performance("Basic 2D Convolution", N, end - start);

        start = std::chrono::high_resolution_clock::now();
        accelerate_convolution(image, processed_image);
        end = std::chrono::high_resolution_clock::now();
        print_performance("Accelerate 2D Convolution", N, end - start);
    }

    return 0;
}
