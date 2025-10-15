#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <complex>

#include <Eigen/Dense>
#include <fftw3.h>
#include "gnuplot-iostream.h"

// Function to read data from a file into a vector
std::vector<double> read_data(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }
    std::vector<double> data;
    double value;
    while (file >> value) {
        data.push_back(value);
    }
    file.close();
    return data;
}

// Function to perform polynomial regression and return the de-trended signal
std::vector<double> perform_regression(const std::vector<double>& y_values, double dt, Eigen::VectorXd& coefficients) {
    int n = y_values.size();
    Eigen::MatrixXd X(n, 4);
    Eigen::VectorXd Y(n);

    for (int i = 0; i < n; ++i) {
        double t = i * dt;
        Y(i) = y_values[i];
        X(i, 0) = 1.0;
        X(i, 1) = t;
        X(i, 2) = t * t;
        X(i, 3) = t * t * t;
    }

    // Solve the least-squares problem using SVD for numerical stability
    coefficients = X.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);

    Eigen::VectorXd y_trend = X * coefficients;

    std::vector<double> y_detrended(n);
    for (int i = 0; i < n; ++i) {
        y_detrended[i] = Y(i) - y_trend(i);
    }
    return y_detrended;
}

// Function to run FFT on the data
std::vector<std::complex<double>> run_fft(const std::vector<double>& data) {
    int n = data.size();
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (n / 2 + 1));
    double* in = (double*)fftw_malloc(sizeof(double) * n);

    // Copy data to fftw-compatible input array
    for(int i = 0; i < n; ++i) {
        in[i] = data[i];
    }

    fftw_plan plan = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    std::vector<std::complex<double>> result(n / 2 + 1);
    for (int i = 0; i < (n / 2 + 1); ++i) {
        result[i] = std::complex<double>(out[i][0], out[i][1]);
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    return result;
}

// Function to find peaks in the magnitude spectrum
std::vector<int> find_peaks(const std::vector<double>& magnitudes, double threshold_ratio) {
    std::vector<int> peak_indices;
    if (magnitudes.size() < 3) {
        return peak_indices;
    }

    double max_magnitude = *std::max_element(magnitudes.begin(), magnitudes.end());
    double threshold = max_magnitude * threshold_ratio;

    for (size_t i = 1; i < magnitudes.size() - 1; ++i) {
        if (magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1] && magnitudes[i] > threshold) {
            peak_indices.push_back(i);
        }
    }
    return peak_indices;
}

// Function to plot the spectrum using gnuplot-iostream
void plot_spectrum(const std::vector<double>& frequencies, const std::vector<double>& magnitudes) {
    Gnuplot gp;
    gp << "set title 'Frequency Spectrum'\n";
    gp << "set xlabel 'Frequency (Hz)'\n";
    gp << "set ylabel 'Magnitude'\n";
    //gp << "set xrange [0:10]\n"; // Focus on the relevant frequency range
    gp << "set grid\n";
    gp << "plot '-' with lines title 'Magnitude'\n";

    std::vector<std::pair<double, double>> data_to_plot;
    for(size_t i = 0; i < frequencies.size(); ++i) {
        data_to_plot.push_back(std::make_pair(frequencies[i], magnitudes[i]));
    }
    gp.send1d(data_to_plot);
}


int main() {
    // --- Parameters ---
    const std::string filename = "f20.txt";
    const double T = 5.0;       // Total observation time
    const double dt = 0.01;     // Time step
    const int N = 501;          // Number of samples

    // --- 1. Data Ingestion and Pre-processing ---
    std::vector<double> y_observed = read_data(filename);
    if (y_observed.size()!= N) {
        std::cerr << "Warning: Expected " << N << " data points, but found " << y_observed.size() << ".\n";
        // Adjust N if necessary, or handle error
    }

    Eigen::VectorXd poly_coeffs;
    std::vector<double> y_detrended = perform_regression(y_observed, dt, poly_coeffs);

    std::cout << "--- Polynomial Trend Analysis ---\n";
    std::cout << "Fitted cubic polynomial: y(t) = a3*t^3 + a2*t^2 + a1*t + a0\n";
    std::cout << "a3 (t^3 coeff): " << poly_coeffs(3) << std::endl;
    std::cout << "a2 (t^2 coeff): " << poly_coeffs(2) << std::endl;
    std::cout << "a1 (t^1 coeff): " << poly_coeffs(1) << std::endl;
    std::cout << "a0 (const coeff): " << poly_coeffs(0) << std::endl;
    std::cout << "---------------------------------\n\n";

    // --- 2. FFT ---
    std::vector<std::complex<double>> fft_result = run_fft(y_detrended);

    // --- 3. Spectral Analysis ---
    int fft_size = fft_result.size();
    std::vector<double> magnitudes(fft_size);
    std::vector<double> frequencies(fft_size);
    double df = 1.0 / T;

    for (int i = 0; i < fft_size; ++i) {
        magnitudes[i] = std::abs(fft_result[i]);
        frequencies[i] = i * df;
    }

    std::vector<int> peak_indices = find_peaks(magnitudes, 0.1); // Threshold: 10% of max peak

    std::cout << "--- Frequency Component Analysis ---\n";
    std::cout << "Identified significant frequencies:\n";
    for (int index : peak_indices) {
        double freq = frequencies[index];
        double mag = magnitudes[index];
        double amplitude = (2.0 * mag) / N; // Amplitude for k > 0
        std::cout << "Frequency: " << freq << " Hz, Magnitude: " << mag << ", Amplitude: " << amplitude << std::endl;
    }
    std::cout << "----------------------------------\n\n";

    // --- 4. Visualization ---
    std::cout << "Generating frequency spectrum plot...\n";
    plot_spectrum(frequencies, magnitudes);
    std::cout << "Plot window generated. Close the window to exit.\n";

    return 0;
}