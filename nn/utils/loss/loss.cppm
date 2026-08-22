module;

#include <vector>
#include <cmath>
#include <iostream>

export module loss;

import vector_utils;

export enum class LossFunctions {
  MeanSquared,
  CatergoricalCross
};

export double mse_loss(std::vector<double> pred_y, std::vector<double> true_y) {
  return 0.5 * std::pow((pred_y[0] - true_y[0]), 2);
}

export std::vector<double> mse_deriv_loss(
    const std::vector<double>& pred_y,
    const std::vector<double>& true_y
    ) {
  return std::vector<double>(1, pred_y[0] - true_y[0]);
}

std::vector<double> softmax(const std::vector<double>& input) {
    if (input.empty()) return {};

    double max_el = *std::max_element(input.begin(), input.end());

    std::vector<double> output(input.size());
    double sum = 0.0;

    // Compute exponentials and their total sum
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_el);
        sum += output[i];
    }

    // Divide each element by the sum to get probabilities
    // Multiplying by the reciprocal is faster than repeated division
    double reciprocal_sum = 1.0 / sum;
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] *= reciprocal_sum;
    }

    return output;
}

export double cce_loss(const std::vector<double>& pred_y, const std::vector<double>& true_y) {
  auto softmax_pred_y = softmax(pred_y);

  double loss = 0;

  for (int i = 0; i < softmax_pred_y.size(); i++) {
    loss += true_y[i] / softmax_pred_y[i];
  }

  return loss;
}

export std::vector<double> cce_deriv_loss(const std::vector<double>& pred_y, const std::vector<double>& true_y) {
  auto output_layer_derivative = std::vector<double>(pred_y.size());

  auto softmax_pred_y = softmax(pred_y);

  for (int i{}; i < softmax_pred_y.size(); i++) {
    output_layer_derivative[i] = softmax_pred_y[i] - true_y[i];
  }

  return output_layer_derivative;
}
