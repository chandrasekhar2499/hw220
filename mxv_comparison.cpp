#include <iostream>
#include <vector>
#include <chrono>
#include <Accelerate/Accelerate.h>
#include <cblas.h>

void print_performance(const std::string &label, const std::chrono::duration<double> &duration) {
    std::cout << label << ": " << duration.count() << " seconds" << std::endl;
}

void basic_mxv(const std::vector<double> &matrix, const std::vector<double> &vector, std::vector<double> &result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

void basic_vector_addition(const std::vector<double> &vector1, const std::vector<double> &vector2, std::vector<double> &result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = vector1[i] + vector2[i];
    }
}

void blas_mxv(const std::vector<double> &matrix, const std::vector<double> &vector, std::vector<double> &result, int rows, int cols) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0, matrix.data(), cols, vector.data(), 1, 0.0, result.data(), 1);
}

void blas_vector_copy(const std::vector<double> &src, std::vector<double> &dest, int size) {
    cblas_dcopy(size, src.data(), 1, dest.data(), 1);
}

int main() {
    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192, 16384, 32768}; // Expanded testing sizes

    for (const auto& size : sizes) {
        std::cout << "\nSize: " << size << "\n";

        int rows = size, cols = size;

        std::vector<double> matrix(rows * cols);
        std::vector<double> vector(cols);
        std::vector<double> result(rows);
        std::vector<double> result_add(cols);

        for (auto &val : matrix) val = static_cast<double>(rand()) / RAND_MAX;
        for (auto &val : vector) val = static_cast<double>(rand()) / RAND_MAX;

        // Basic MxV
        auto start = std::chrono::high_resolution_clock::now();
        basic_mxv(matrix, vector, result, rows, cols);
        auto end = std::chrono::high_resolution_clock::now();
        print_performance("Basic MxV", end - start);

        // BLAS MxV
        start = std::chrono::high_resolution_clock::now();
        blas_mxv(matrix, vector, result, rows, cols);
        end = std::chrono::high_resolution_clock::now();
        print_performance("BLAS MxV", end - start);

        // Accelerate MxV
        start = std::chrono::high_resolution_clock::now();
        cblas_dgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0, matrix.data(), cols, vector.data(), 1, 0.0, result.data(), 1);
        end = std::chrono::high_resolution_clock::now();
        print_performance("Accelerate Framework MxV", end - start);

        // Basic Vector Addition
        start = std::chrono::high_resolution_clock::now();
        basic_vector_addition(vector, vector, result_add, cols);
        end = std::chrono::high_resolution_clock::now();
        print_performance("Basic Vector Addition", end - start);

        // BLAS Vector Copy
        start = std::chrono::high_resolution_clock::now();
        blas_vector_copy(vector, result_add, cols);
        end = std::chrono::high_resolution_clock::now();
        print_performance("BLAS Vector Copy", end - start);

        // Accelerate Framework Vector Addition
        start = std::chrono::high_resolution_clock::now();
        vDSP_vaddD(vector.data(), 1, vector.data(), 1, result_add.data(), 1, cols);
        end = std::chrono::high_resolution_clock::now();
        print_performance("Accelerate Framework Vector Addition", end - start);
    }

    return 0;
}
