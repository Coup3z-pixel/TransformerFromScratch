module;

#include <vector>

export module loss;

export enum class LossFunctions {
  MeanSquared,
  CatergoricalCross
};

export double mse_loss(std::vector<double> pred_y, std::vector<double> true_y) {
  return 0.0;
}

export std::vector<double> mse_deriv_loss(std::vector<double>, std::vector<double>) {
  return std::vector<double>();
}

export double cce_loss(std::vector<double> pred_y, std::vector<double> true_y) {
  return 0.0;
}

export std::vector<double> cce_deriv_loss(std::vector<double> pred_y, std::vector<double> true_y) {
  return std::vector<double>();
}
