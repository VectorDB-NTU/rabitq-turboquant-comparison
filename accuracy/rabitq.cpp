#include <sys/types.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/rotator.hpp"

using data_type = rabitqlib::RowMajorArray<float>;

int main(int argc, char** argv) {
    assert(argc == 3);

    data_type train;
    data_type test;
    rabitqlib::load_vecs<float, data_type>("./dbpedia-openai3-1536-train.fvecs", train);
    rabitqlib::load_vecs<float, data_type>("./dbpedia-openai3-1536-test.fvecs", test);

    size_t dim = train.cols();
    size_t bit = atoi(argv[1]);
    std::string type = argv[2];
    assert(type == "mse" || type == "prod");

    auto quant_type = rabitqlib::ScalarQuantizerType::UNBIASED_ESTIMATION;
    if (type == "mse") {
        std::cout << "Using MSE-optimized quantization\n";
        quant_type = rabitqlib::ScalarQuantizerType::RECONSTRUCTION;
    } else {
        std::cout << "Using Product-optimized quantization\n";
    }

    // choose rotator
    auto *rotator = new rabitqlib::rotator_impl::MatrixRotator<float>(dim, dim);

    auto rotated_data = data_type(train.rows(), train.cols());
    auto rotated_test = data_type(test.rows(), test.cols());

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < train.rows(); ++i) {
        rotator->rotate(&train(i, 0), &rotated_data(i, 0));
    }
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < test.rows(); ++i) {
        rotator->rotate(&test(i, 0), &rotated_test(i, 0));
    }

    auto config = rabitqlib::quant::RabitqConfig();

    auto reconstructed_data = data_type(train.rows(), train.cols());
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < rotated_data.rows(); ++i) {
        auto code = std::vector<uint16_t>(dim);
        float delta = 0;
        float vl = 0;
        rabitqlib::quant::quantize_scalar(
            &rotated_data(i, 0),
            dim,
            bit,
            code.data(),
            delta,
            vl,
            config,
            quant_type
        );
        rabitqlib::quant::reconstruct_vec(
            code.data(), delta, vl, dim, &reconstructed_data(i, 0)
        );
    }

    // Compute per-pair IP error: orig_ip[j,i] - recon_ip[j,i], flattened
    size_t n_test = rotated_test.rows();
    size_t n_data = rotated_data.rows();
    std::vector<float> ip_errors(n_test * n_data);
#pragma omp parallel for schedule(dynamic)
    for (size_t j = 0; j < n_test; ++j) {
        for (size_t i = 0; i < n_data; ++i) {
            double ip_orig = 0.0;
            double ip_recon = 0.0;
            for (size_t k = 0; k < dim; ++k) {
                ip_orig += (double)rotated_data(i, k) * rotated_test(j, k);
                ip_recon += (double)reconstructed_data(i, k) * rotated_test(j, k);
            }
            ip_errors[j * n_data + i] = static_cast<float>(ip_orig - ip_recon);
        }
    }

    std::string ip_fname = "rabitq_ip_errors_" + std::to_string(bit) + "bit_" + type + ".csv";

    // Write ip_errors to CSV (one value per line)
    {
        std::ofstream out(ip_fname);
        out << "ip_error\n";
        for (float ip_error : ip_errors) {
            out << ip_error << "\n";
        }
    }

    delete rotator;
    return 0;
}
