#include <iostream>
#include <cassert>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>
#include <ranges>

#define MNIST_LABEL_AMOUNTS 10

import mnist;
import nn.Network;
import nn.ActivationFunction;
import nn.layers.adjustable.HiddenLayer;
import nn.layers.adjustable.Adjustable;


std::vector<std::vector<double>> one_state_encode_labels(const std::vector<uint8_t> raw_labels) {
  auto one_state_encoded_labels = std::vector<std::vector<double>>(raw_labels.size());

  for (auto&& [index, label] : std::views::enumerate(raw_labels)) {
    one_state_encoded_labels[index] = std::vector<double>(MNIST_LABEL_AMOUNTS);
    one_state_encoded_labels[index][label] = 1.0;
  }

  return one_state_encoded_labels;
}

int main() {
  Network neural_network = Network(784, {
      HiddenLayer(512, ActivationFunction::relu),
      HiddenLayer(256, ActivationFunction::relu),
      HiddenLayer(128, ActivationFunction::relu),
      HiddenLayer(64, ActivationFunction::relu),
      HiddenLayer(MNIST_LABEL_AMOUNTS, ActivationFunction::relu),
      }, LossFunctions::CatergoricalCross);

  auto [X_train, raw_Y_train] = load_mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
  auto [X_test, raw_Y_test] = load_mnist("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

  auto Y_train = one_state_encode_labels(raw_Y_train);
  auto Y_test = one_state_encode_labels(raw_Y_test);

  neural_network.train(
      X_train,
      Y_train,
      X_test,
      Y_test,
      20,
      0.001
    );

  return 0;
}
