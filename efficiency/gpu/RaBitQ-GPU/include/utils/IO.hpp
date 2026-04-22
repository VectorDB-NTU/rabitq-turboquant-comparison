#pragma once

#include <stdint.h>
#include <sys/stat.h>

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

inline size_t get_filesize(const char* filename) {
    struct stat64 stat_buf;
    int rc = stat64(filename, &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

inline bool file_exits(const char* filename) {
    std::ifstream f(filename);
    if (!f.good()) {
        f.close();
        return false;
    }
    f.close();
    return true;
}

template <typename T, class M>
void load_vecs(const char* filename, M& Mat) {
    if (!file_exits(filename)) {
        std::cerr << "File " << filename << " not exists\n";
        abort();
    }

    uint32_t tmp;
    size_t file_size = get_filesize(filename);
    std::ifstream input(filename, std::ios::binary);

    input.read((char*)&tmp, sizeof(uint32_t));

    size_t cols = tmp;
    size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
    Mat = M(rows, cols);

    input.seekg(0, input.beg);

    for (size_t i = 0; i < rows; i++) {
        input.read((char*)&tmp, sizeof(uint32_t));
        input.read((char*)&Mat(i, 0), sizeof(T) * cols);
    }

    std::cout << "File " << filename << " loaded\n";
    std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
    input.close();
}
