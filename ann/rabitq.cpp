#include <omp.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/rotator.hpp"

using PID = rabitqlib::PID;
using data_type = rabitqlib::RowMajorArray<float>;

namespace {

struct Options {
    std::string base_file;
    std::string query_file;
    std::string neighbors_file;
    size_t total_bits = 0;
    rabitqlib::MetricType metric_type = rabitqlib::METRIC_IP;
    bool faster_quant = false;
    size_t query_limit = 0;
    int num_threads = 0;
    std::vector<size_t> recall_ranks = {1, 2, 4, 8, 16, 32, 64};
    std::string rotator = "auto";
};

struct ScoredPID {
    float score;
    PID id;
};

struct MinScoreFirst {
    bool operator()(const ScoredPID& lhs, const ScoredPID& rhs) const {
        return lhs.score > rhs.score;
    }
};

struct DistPID {
    float dist;
    PID id;
};

struct MaxDistFirst {
    bool operator()(const DistPID& lhs, const DistPID& rhs) const { return lhs.dist < rhs.dist; }
};

struct Int32NpyMatrix {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<int32_t> data;

    [[nodiscard]] const int32_t* row(size_t idx) const { return data.data() + (idx * cols); }
};

std::string trim(std::string value) {
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(
        std::find_if(value.rbegin(), value.rend(), not_space).base(),
        value.end()
    );
    return value;
}

bool parse_bool(const std::string& value) {
    if (value == "true" || value == "True" || value == "1") {
        return true;
    }
    if (value == "false" || value == "False" || value == "0") {
        return false;
    }
    throw std::invalid_argument("Invalid boolean value: " + value);
}

std::vector<size_t> parse_ranks(const std::string& spec) {
    std::vector<size_t> ranks;
    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        const size_t rank = static_cast<size_t>(std::stoul(token));
        if (rank == 0) {
            throw std::invalid_argument("Recall ranks must be positive");
        }
        ranks.push_back(rank);
    }
    if (ranks.empty()) {
        throw std::invalid_argument("Recall ranks list is empty");
    }
    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());
    return ranks;
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " --base <train.fvecs> --query <test.fvecs> --bits <total_bits>\n"
              << "       [--metric ip|l2] [--faster-quant true|false]\n"
              << "       [--neighbors <groundtruth_neighbors.npy>] [--query-limit <n>]\n"
              << "       [--num-threads <n>] [--ranks 1,2,4,8,16,32,64]\n"
              << "       [--rotator auto|identity|matrix|fht]\n\n"
              << "This evaluator uses full-bit codes for every point directly, without IVF\n"
              << "1-bit pruning, and keeps the rotated dimension equal to the original dim.\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto next_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--base") {
            opts.base_file = next_value(arg);
        } else if (arg == "--query") {
            opts.query_file = next_value(arg);
        } else if (arg == "--bits") {
            opts.total_bits = static_cast<size_t>(std::stoul(next_value(arg)));
        } else if (arg == "--metric") {
            const std::string value = next_value(arg);
            if (value == "ip" || value == "IP") {
                opts.metric_type = rabitqlib::METRIC_IP;
            } else if (value == "l2" || value == "L2") {
                opts.metric_type = rabitqlib::METRIC_L2;
            } else {
                throw std::invalid_argument("Unsupported metric: " + value);
            }
        } else if (arg == "--faster-quant") {
            opts.faster_quant = parse_bool(next_value(arg));
        } else if (arg == "--neighbors") {
            opts.neighbors_file = next_value(arg);
        } else if (arg == "--query-limit") {
            opts.query_limit = static_cast<size_t>(std::stoul(next_value(arg)));
        } else if (arg == "--num-threads") {
            opts.num_threads = std::stoi(next_value(arg));
        } else if (arg == "--ranks") {
            opts.recall_ranks = parse_ranks(next_value(arg));
        } else if (arg == "--rotator") {
            opts.rotator = next_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    if (opts.base_file.empty() || opts.query_file.empty() || opts.total_bits == 0) {
        throw std::invalid_argument("base/query/bits are required");
    }
    if (opts.total_bits > 8) {
        throw std::invalid_argument(
            "This evaluator currently supports total_bits in [1, 8] because it stores "
            "full-bit codes as uint8_t"
        );
    }
    return opts;
}

Int32NpyMatrix load_int32_npy_matrix(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open NPY file: " + path);
    }

    char magic[6];
    input.read(magic, sizeof(magic));
    if (std::memcmp(magic, "\x93NUMPY", sizeof(magic)) != 0) {
        throw std::runtime_error("Invalid NPY magic for: " + path);
    }

    uint8_t major = 0;
    uint8_t minor = 0;
    input.read(reinterpret_cast<char*>(&major), sizeof(major));
    input.read(reinterpret_cast<char*>(&minor), sizeof(minor));

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        input.read(reinterpret_cast<char*>(&len16), sizeof(len16));
        header_len = len16;
    } else if (major == 2) {
        input.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    } else {
        throw std::runtime_error("Unsupported NPY version for: " + path);
    }

    std::string header(header_len, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header_len));

    if (header.find("'fortran_order': False") == std::string::npos &&
        header.find("\"fortran_order\": False") == std::string::npos) {
        throw std::runtime_error("Only C-order NPY matrices are supported: " + path);
    }

    if (header.find("'descr': '<i4'") == std::string::npos &&
        header.find("\"descr\": \"<i4\"") == std::string::npos &&
        header.find("'descr': '|i4'") == std::string::npos) {
        throw std::runtime_error("Only int32 NPY matrices are supported: " + path);
    }

    const size_t shape_begin = header.find('(');
    const size_t shape_end = header.find(')', shape_begin);
    if (shape_begin == std::string::npos || shape_end == std::string::npos) {
        throw std::runtime_error("Failed to parse NPY shape: " + path);
    }

    std::vector<size_t> shape;
    std::stringstream shape_ss(header.substr(shape_begin + 1, shape_end - shape_begin - 1));
    std::string token;
    while (std::getline(shape_ss, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        shape.push_back(static_cast<size_t>(std::stoull(token)));
    }
    if (shape.size() != 2) {
        throw std::runtime_error("Only 2D NPY matrices are supported: " + path);
    }

    Int32NpyMatrix matrix;
    matrix.rows = shape[0];
    matrix.cols = shape[1];
    matrix.data.resize(matrix.rows * matrix.cols);
    input.read(
        reinterpret_cast<char*>(matrix.data.data()),
        static_cast<std::streamsize>(matrix.data.size() * sizeof(int32_t))
    );
    if (!input) {
        throw std::runtime_error("Failed to read NPY payload: " + path);
    }
    return matrix;
}

size_t max_rank(const std::vector<size_t>& ranks) {
    return *std::max_element(ranks.begin(), ranks.end());
}

std::vector<std::vector<PID>> exact_topk_from_neighbors(
    const Int32NpyMatrix& neighbors, size_t nq, size_t eval_rank
) {
    if (neighbors.rows < nq) {
        throw std::runtime_error("neighbors.npy has fewer rows than query count");
    }
    if (neighbors.cols < eval_rank) {
        throw std::runtime_error("neighbors.npy does not contain enough columns for requested ranks");
    }

    std::vector<std::vector<PID>> exact_topk(nq, std::vector<PID>(eval_rank));
    for (size_t i = 0; i < nq; ++i) {
        const int32_t* row = neighbors.row(i);
        for (size_t j = 0; j < eval_rank; ++j) {
            exact_topk[i][j] = static_cast<PID>(row[j]);
        }
    }
    return exact_topk;
}

std::vector<PID> compute_exact_topk_for_query(
    const Eigen::Ref<const Eigen::RowVectorXf>& dot_scores,
    const std::vector<float>& data_sqnorms,
    float query_sqnorm,
    rabitqlib::MetricType metric_type,
    size_t eval_rank
) {
    std::priority_queue<ScoredPID, std::vector<ScoredPID>, MinScoreFirst> heap;

    for (Eigen::Index j = 0; j < dot_scores.cols(); ++j) {
        float score = dot_scores(j);
        if (metric_type == rabitqlib::METRIC_L2) {
            const float dist = query_sqnorm + data_sqnorms[static_cast<size_t>(j)] - 2.0F * score;
            score = -dist;
        }

        if (heap.size() < eval_rank) {
            heap.push({score, static_cast<PID>(j)});
        } else if (score > heap.top().score) {
            heap.pop();
            heap.push({score, static_cast<PID>(j)});
        }
    }

    std::vector<ScoredPID> best;
    best.reserve(eval_rank);
    while (!heap.empty()) {
        best.push_back(heap.top());
        heap.pop();
    }
    std::sort(
        best.begin(),
        best.end(),
        [](const ScoredPID& lhs, const ScoredPID& rhs) { return lhs.score > rhs.score; }
    );

    std::vector<PID> topk(eval_rank);
    for (size_t i = 0; i < eval_rank; ++i) {
        topk[i] = best[i].id;
    }
    return topk;
}

std::vector<std::vector<PID>> compute_exact_topk(
    const data_type& data,
    const data_type& query,
    rabitqlib::MetricType metric_type,
    size_t eval_rank
) {
    std::cout << "Computing exact top" << eval_rank << " ground truth...\n";
    const auto dot_scores = (query.matrix() * data.matrix().transpose()).eval();

    std::vector<float> data_sqnorms;
    std::vector<float> query_sqnorms;
    if (metric_type == rabitqlib::METRIC_L2) {
        data_sqnorms.resize(static_cast<size_t>(data.rows()));
        query_sqnorms.resize(static_cast<size_t>(query.rows()));
        for (Eigen::Index j = 0; j < data.rows(); ++j) {
            data_sqnorms[static_cast<size_t>(j)] = data.row(j).matrix().squaredNorm();
        }
        for (Eigen::Index i = 0; i < query.rows(); ++i) {
            query_sqnorms[static_cast<size_t>(i)] = query.row(i).matrix().squaredNorm();
        }
    }

    std::vector<std::vector<PID>> exact_topk(static_cast<size_t>(query.rows()));
    for (Eigen::Index i = 0; i < query.rows(); ++i) {
        const float query_sqnorm =
            metric_type == rabitqlib::METRIC_L2 ? query_sqnorms[static_cast<size_t>(i)] : 0.0F;
        exact_topk[static_cast<size_t>(i)] = compute_exact_topk_for_query(
            dot_scores.row(i), data_sqnorms, query_sqnorm, metric_type, eval_rank
        );
    }
    return exact_topk;
}

float dot_query_code(const float* query, const uint8_t* code, size_t dim) {
    float sum = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
        sum += query[i] * static_cast<float>(code[i]);
    }
    return sum;
}

std::unique_ptr<rabitqlib::Rotator<float>> build_rotator(size_t dim, const std::string& choice) {
    if (choice == "identity") {
        return nullptr;
    }
    if (choice == "auto") {
        if (dim % 64 == 0) {
            return std::unique_ptr<rabitqlib::Rotator<float>>(
                rabitqlib::choose_rotator<float>(
                    dim, rabitqlib::RotatorType::FhtKacRotator, dim
                )
            );
        }
        return std::unique_ptr<rabitqlib::Rotator<float>>(
            rabitqlib::choose_rotator<float>(
                dim, rabitqlib::RotatorType::MatrixRotator, dim
            )
        );
    }
    if (choice == "matrix") {
        return std::unique_ptr<rabitqlib::Rotator<float>>(
            rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::MatrixRotator, dim)
        );
    }
    if (choice == "fht") {
        if (dim % 64 != 0) {
            throw std::invalid_argument("FHT rotator requires dim to be a multiple of 64");
        }
        return std::unique_ptr<rabitqlib::Rotator<float>>(
            rabitqlib::choose_rotator<float>(dim, rabitqlib::RotatorType::FhtKacRotator, dim)
        );
    }
    throw std::invalid_argument("Unsupported rotator choice: " + choice);
}

data_type maybe_limit_queries(const data_type& query, size_t query_limit) {
    if (query_limit == 0 || query_limit >= static_cast<size_t>(query.rows())) {
        return query;
    }
    data_type limited(query_limit, query.cols());
    limited = query.topRows(static_cast<Eigen::Index>(query_limit));
    return limited;
}

data_type rotate_dataset(
    const data_type& data, const rabitqlib::Rotator<float>* rotator, int num_threads
) {
    data_type rotated(data.rows(), data.cols());
    if (rotator == nullptr) {
        rotated = data;
        return rotated;
    }

#pragma omp parallel for if(num_threads != 1)
    for (Eigen::Index i = 0; i < data.rows(); ++i) {
        rotator->rotate(&data(i, 0), &rotated(i, 0));
    }
    return rotated;
}

std::vector<std::vector<PID>> compute_approx_topk(
    const data_type& rotated_data,
    const data_type& rotated_query,
    const std::vector<uint8_t>& full_codes,
    const std::vector<float>& f_add,
    const std::vector<float>& f_rescale,
    size_t total_bits,
    rabitqlib::MetricType metric_type,
    size_t eval_rank,
    int num_threads
) {
    const size_t num_points = static_cast<size_t>(rotated_data.rows());
    const size_t nq = static_cast<size_t>(rotated_query.rows());
    const size_t dim = static_cast<size_t>(rotated_data.cols());
    std::vector<std::vector<PID>> approx_topk(nq, std::vector<PID>(eval_rank));

#pragma omp parallel for if(num_threads != 1)
    for (Eigen::Index qi = 0; qi < rotated_query.rows(); ++qi) {
        const float* query_ptr = &rotated_query(qi, 0);
        const float query_sum =
            std::accumulate(query_ptr, query_ptr + dim, 0.0F);
        const float kb_sumq =
            query_sum * (-static_cast<float>((1U << total_bits) - 1U) / 2.0F);
        const float g_add =
            metric_type == rabitqlib::METRIC_L2
                ? rotated_query.row(qi).matrix().squaredNorm()
                : 0.0F;

        std::priority_queue<DistPID, std::vector<DistPID>, MaxDistFirst> heap;
        for (size_t j = 0; j < num_points; ++j) {
            const uint8_t* code_ptr = full_codes.data() + (j * dim);
            const float dist =
                f_add[j] + g_add + (f_rescale[j] * (dot_query_code(query_ptr, code_ptr, dim) + kb_sumq));

            if (heap.size() < eval_rank) {
                heap.push({dist, static_cast<PID>(j)});
            } else if (dist < heap.top().dist) {
                heap.pop();
                heap.push({dist, static_cast<PID>(j)});
            }
        }

        std::vector<DistPID> best;
        best.reserve(eval_rank);
        while (!heap.empty()) {
            best.push_back(heap.top());
            heap.pop();
        }
        std::sort(
            best.begin(),
            best.end(),
            [](const DistPID& lhs, const DistPID& rhs) { return lhs.dist < rhs.dist; }
        );
        for (size_t i = 0; i < eval_rank; ++i) {
            approx_topk[static_cast<size_t>(qi)][i] = best[i].id;
        }
    }

    return approx_topk;
}

void print_recall_table(
    const std::vector<std::vector<size_t>>& hits,
    const std::vector<size_t>& recall_ranks,
    size_t nq
) {
    std::cout << "t\\K";
    for (size_t k : recall_ranks) {
        std::cout << '\t' << k;
    }
    std::cout << '\n';

    for (size_t t_idx = 0; t_idx < recall_ranks.size(); ++t_idx) {
        const size_t t = recall_ranks[t_idx];
        std::cout << t;
        for (size_t k_idx = 0; k_idx < recall_ranks.size(); ++k_idx) {
            const float recall =
                static_cast<float>(hits[t_idx][k_idx]) / static_cast<float>(nq * t);
            std::cout << '\t' << recall;
        }
        std::cout << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        if (opts.num_threads > 0) {
            omp_set_num_threads(opts.num_threads);
        }

        data_type data;
        data_type query;
        rabitqlib::load_vecs<float, data_type>(opts.base_file.c_str(), data);
        rabitqlib::load_vecs<float, data_type>(opts.query_file.c_str(), query);
        query = maybe_limit_queries(query, opts.query_limit);

        if (data.cols() != query.cols()) {
            throw std::runtime_error("Base/query dimensions mismatch");
        }

        const size_t dim = static_cast<size_t>(data.cols());
        const size_t num_points = static_cast<size_t>(data.rows());
        const size_t nq = static_cast<size_t>(query.rows());
        const size_t eval_rank = max_rank(opts.recall_ranks);

        std::cout << "Base/query loaded\n";
        std::cout << "\tbase: " << opts.base_file << '\n';
        std::cout << "\tquery: " << opts.query_file << '\n';
        if (!opts.neighbors_file.empty()) {
            std::cout << "\tneighbors: " << opts.neighbors_file << '\n';
        }
        std::cout << "\tN: " << num_points << '\n';
        std::cout << "\tQ: " << nq << '\n';
        std::cout << "\tDIM: " << dim << " (no padding)\n";
        std::cout << "\tmetric: "
                  << (opts.metric_type == rabitqlib::METRIC_IP ? "IP" : "L2") << '\n';
        std::cout << "\ttotal_bits: " << opts.total_bits << '\n';
        std::cout << "\tfaster_quant: " << (opts.faster_quant ? "true" : "false") << '\n';
        std::cout << "\trotator: " << opts.rotator << '\n';
        std::cout << "\tthreads: " << omp_get_max_threads() << '\n';

        const auto rotator = build_rotator(dim, opts.rotator);
        const data_type rotated_data = rotate_dataset(data, rotator.get(), opts.num_threads);
        const data_type rotated_query = rotate_dataset(query, rotator.get(), opts.num_threads);

        std::cout << "Quantizing full-bit codes without pruning...\n";
        std::vector<uint8_t> full_codes(num_points * dim, 0);
        std::vector<float> f_add(num_points, 0.0F);
        std::vector<float> f_rescale(num_points, 0.0F);
        std::vector<float> f_error(num_points, 0.0F);

        const auto config = opts.faster_quant
                                ? rabitqlib::quant::faster_config(dim, opts.total_bits)
                                : rabitqlib::quant::RabitqConfig();

#pragma omp parallel for if(opts.num_threads != 1)
        for (Eigen::Index i = 0; i < rotated_data.rows(); ++i) {
            rabitqlib::quant::quantize_full_single<float, uint8_t>(
                &rotated_data(i, 0),
                dim,
                opts.total_bits,
                full_codes.data() + (static_cast<size_t>(i) * dim),
                f_add[static_cast<size_t>(i)],
                f_rescale[static_cast<size_t>(i)],
                f_error[static_cast<size_t>(i)],
                opts.metric_type,
                config
            );
        }

        std::vector<std::vector<PID>> exact_topk;
        if (!opts.neighbors_file.empty()) {
            const auto neighbors = load_int32_npy_matrix(opts.neighbors_file);
            exact_topk = exact_topk_from_neighbors(neighbors, nq, eval_rank);
        } else {
            exact_topk = compute_exact_topk(data, query, opts.metric_type, eval_rank);
        }

        const auto approx_topk = compute_approx_topk(
            rotated_data,
            rotated_query,
            full_codes,
            f_add,
            f_rescale,
            opts.total_bits,
            opts.metric_type,
            eval_rank,
            opts.num_threads
        );

        std::vector<std::vector<size_t>> hits(
            opts.recall_ranks.size(), std::vector<size_t>(opts.recall_ranks.size(), 0)
        );
        for (size_t i = 0; i < nq; ++i) {
            for (size_t t_idx = 0; t_idx < opts.recall_ranks.size(); ++t_idx) {
                const size_t t = opts.recall_ranks[t_idx];
                for (size_t k_idx = 0; k_idx < opts.recall_ranks.size(); ++k_idx) {
                    const size_t topk = opts.recall_ranks[k_idx];
                    size_t overlap = 0;
                    for (size_t exact_idx = 0; exact_idx < t; ++exact_idx) {
                        const PID exact_id = exact_topk[i][exact_idx];
                        for (size_t approx_idx = 0; approx_idx < topk; ++approx_idx) {
                            if (approx_topk[i][approx_idx] == exact_id) {
                                overlap += 1;
                                break;
                            }
                        }
                    }
                    hits[t_idx][k_idx] += overlap;
                }
            }
        }

        print_recall_table(hits, opts.recall_ranks, nq);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}
