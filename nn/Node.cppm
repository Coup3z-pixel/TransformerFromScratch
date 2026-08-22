module;

#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>

export module nn.Node;

export enum class ActivationFunction {
  relu,
  sigmoid
};


export std::vector<double> initialize_random_weights(int num_of_input_nodes) {
  static std::random_device rd;
  static std::mt19937 gen(rd());

  double range = std::sqrt(6.0 / static_cast<double>(num_of_input_nodes));

  std::vector<double> random_weights(num_of_input_nodes);
  std::uniform_real_distribution<double> distrib(-range, range);

  for (int i = 0; i < num_of_input_nodes; i++) {
    random_weights[i] = distrib(gen);
  }

  return random_weights;
}

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

export class Node {
  private:
    std::vector<double> weights;
    double bias;
    ActivationFunction activation_func;

  public:
    Node(int prev_layer_node_num, ActivationFunction activation_function, int node_pos = -1, int layer_pos = -1) {
      if (node_pos != -1) {
        std::cout << "Node: " << (node_pos + 1) << " in Layer: " << layer_pos << "\n";
      }

      weights = initialize_random_weights(prev_layer_node_num);

      activation_func = activation_function;
      bias = initialize_random_weights(1)[0];
    }

    Node() = default;

    double process(std::vector<double>& input) {
      return std::inner_product(weights.begin(), weights.end(), input.begin(), 0.0) + bias;
    }

    std::vector<double>& get_weights() {
      return weights;
    }

    int get_num_of_weights() {
      return weights.size();
    }


    void adjust_bias(double delta) {
      bias -= delta;
    }

    void adjust_weight(int index, double delta) {
      weights[index] -= delta;
    }
};
