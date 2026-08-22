module;

#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>

export module nn.ActivationFunction;

export enum class ActivationFunction {
  relu,
  sigmoid
};

export double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

export double sigmoid_derivative(double activated_val) {
    return activated_val * (1.0 - activated_val);
}

export double relu(double x) {
    return std::max(0.0, x);
}

export double relu_derivative(double activated_val) {
    if (activated_val <= 0.0) {
        return 0.001;
    }
    return 1.0;
}
