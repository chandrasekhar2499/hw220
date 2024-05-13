# hw220
homework files for cpp acceleration

steps to runL

g++ mxv_comparison.cpp -O2 -std=c++11 -DACCELERATE_NEW_LAPACK \
    -framework Accelerate \
    -I/opt/homebrew/opt/openblas/include \
    -L/opt/homebrew/opt/openblas/lib \
    -lopenblas -o mxv_comparison


./mxv_comparison 


clang++ -std=c++11 -o fft_cnn fft_cnn.cpp $(pkg-config --cflags --libs opencv4) -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3 -framework Accelerate

./fft_cnn