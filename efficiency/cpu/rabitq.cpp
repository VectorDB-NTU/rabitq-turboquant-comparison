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
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/rotator.hpp"

using data_type = rabitqlib::RowMajorArray<float>;

int main(int argc, char** argv) {
    assert(argc == 4);
    
    std::string train_fname = argv[1];
    size_t bit = atoi(argv[2]);
    std::string fast_quant = argv[3];
    assert(fast_quant == "true" || fast_quant == "false");

    data_type train;
    rabitqlib::load_vecs<float, data_type>(train_fname.data(), train);

    size_t dim = train.cols();

    rabitqlib::Rotator<float> *rotator = nullptr;
    // choose rotator
    if (fast_quant == "true") {
        std::cout << "Using fast quantization with FHT Kac rotator\n";
        rotator = new rabitqlib::rotator_impl::FhtKacRotator(dim, dim);

    } else {
        std::cout << "Using normal quantization with matrix rotator\n";
        rotator = new rabitqlib::rotator_impl::MatrixRotator<float>(dim, dim);
    }

    auto rotated_data = data_type(train.rows(), train.cols());

    rabitqlib::StopW stopw;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < train.rows(); ++i) {
        rotator->rotate(&train(i, 0), &rotated_data(i, 0));
    }

    auto config = rabitqlib::quant::RabitqConfig();
    if (fast_quant == "true") {
        config = rabitqlib::quant::faster_config(dim, bit);
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < rotated_data.rows(); ++i) {
        auto code = std::vector<uint8_t>(dim);
        float delta = 0;
        float vl = 0;
        rabitqlib::quant::quantize_scalar(
            &rotated_data(i, 0),
            dim,
            bit,
            code.data(),
            delta,
            vl,
            config
        );
    }

    auto time = stopw.get_elapsed_mili();
    std::cout << "Quantization time for " << bit << " bits (Using fast quant " + fast_quant + ") : " << time / 1000 << " seconds\n";

    return 0;
}