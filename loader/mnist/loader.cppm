module;

#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

export module mnist;

export struct MnistData {
    std::vector<std::vector<double>> images; // [num_samples][784]
    std::vector<uint8_t> raw_labels;            // [num_samples]
};

// MNIST files store 32-bit integers in Big-Endian format.
// This swaps bytes to Little-Endian for x86/ARM architectures.
static uint32_t swap_bytes(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

export MnistData load_mnist(const std::string& image_path, const std::string& label_path) {
    // 1. Read Images File Header
    std::ifstream img_file(image_path, std::ios::binary);
    if (!img_file.is_open()) {
        throw std::runtime_error("Failed to open image file: " + image_path);
    }

    uint32_t img_magic = 0, num_images = 0, rows = 0, cols = 0;
    img_file.read(reinterpret_cast<char*>(&img_magic), 4);
    img_file.read(reinterpret_cast<char*>(&num_images), 4);
    img_file.read(reinterpret_cast<char*>(&rows), 4);
    img_file.read(reinterpret_cast<char*>(&cols), 4);

    img_magic  = swap_bytes(img_magic);
    num_images = swap_bytes(num_images);
    rows       = swap_bytes(rows);
    cols       = swap_bytes(cols);

    if (img_magic != 2051) { // 2051 is the magic number for MNIST images
        throw std::runtime_error("Invalid magic number in image file: " + image_path);
    }

    // 2. Read Labels File Header
    std::ifstream lbl_file(label_path, std::ios::binary);
    if (!lbl_file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + label_path);
    }

    uint32_t lbl_magic = 0, num_labels = 0;
    lbl_file.read(reinterpret_cast<char*>(&lbl_magic), 4);
    lbl_file.read(reinterpret_cast<char*>(&num_labels), 4);

    lbl_magic  = swap_bytes(lbl_magic);
    num_labels = swap_bytes(num_labels);

    if (lbl_magic != 2049) { // 2049 is the magic number for MNIST labels
        throw std::runtime_error("Invalid magic number in label file: " + label_path);
    }

    if (num_images != num_labels) {
        throw std::runtime_error("Image and label count mismatch.");
    }

    // 3. Extract Payload Data
    const size_t image_size = rows * cols; // 28 * 28 = 784
    MnistData dataset;
    dataset.images.resize(num_images, std::vector<double>(image_size));
    dataset.raw_labels.resize(num_labels);

    std::vector<uint8_t> buffer(image_size);
    for (size_t i = 0; i < num_images; ++i) {
        img_file.read(reinterpret_cast<char*>(buffer.data()), image_size);
        for (size_t j = 0; j < image_size; ++j) {
            dataset.images[i][j] = static_cast<double>(buffer[j]) / 255.0f; // Scale to [0.0, 1.0]
        }
    }

    lbl_file.read(reinterpret_cast<char*>(dataset.raw_labels.data()), num_labels);

    return dataset;
}
