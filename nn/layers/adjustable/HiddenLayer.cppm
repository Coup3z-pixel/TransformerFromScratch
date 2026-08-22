module;

#include <vector>
#include <iostream>
#include <functional>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

export module nn.layers.adjustable.HiddenLayer;

import nn.ActivationFunction;
import nn.layers.adjustable.Adjustable;
import nn.layers.adjustable.utils;

export class HiddenLayer : public AdjustableLayer
{
  private:
    ActivationFunction activation_func_type;
    int num_of_nodes;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    std::function<double(double)> activation_function;
    std::function<double(double)> activation_derivative;

    float layer_dropout;

  public:
    HiddenLayer(
        int layer_node_count,
        ActivationFunction activ_func_type = ActivationFunction::relu,
        float dropout = 0.0
        ) {
      num_of_nodes = layer_node_count;
      layer_dropout = dropout;

      activation_func_type = activ_func_type;
      switch (activation_func_type) {
        case ActivationFunction::relu:
          activation_function = relu;
          activation_derivative = relu_derivative;
          break;
        case ActivationFunction::sigmoid:
          activation_function = sigmoid;
          activation_derivative = sigmoid_derivative;
          break;
        default:
          break;
      }
    }

    HiddenLayer() = default;

    int get_num_of_nodes() {
      return num_of_nodes;
    }

    int get_num_of_weights() {
      return weights[0].size();
    }

    std::vector<std::vector<double>> get_layer_weights() {
      return weights;
    }

    ActivationFunction get_activation_func() {
      return activation_func_type;
    }

    std::function<double(double)> get_activation_function() {
      return activation_function;
    }

    std::function<double(double)> get_activation_derivative() {
      return activation_derivative;
    }

    void adjust_bias(int node_idx, double adjust_factor) {
      biases[node_idx] -= adjust_factor;
    }

    void adjust_nodes_weight(int node_idx, int weight_idx, double adjust_factor) {
      weights[node_idx][weight_idx] -= adjust_factor;
    }

    void setup_layer(int prev_layer_node_num, int layer_idx = -1) {
      if (layer_idx != -1) {
        std::cout << "Init Layer " << layer_idx << " with " << prev_layer_node_num << " previous nodes \n";
      }

      weights = std::vector<std::vector<double>>(num_of_nodes);
      biases = initialize_random_weights(num_of_nodes);

      for (int n = 0; n < num_of_nodes; n++) {
        weights[n] = initialize_random_weights(prev_layer_node_num);
      }
    }

    std::vector<double> calculate_layer(const std::vector<double>& prev_layer) {
      std::vector<double> curr_layer = std::vector<double>(num_of_nodes);

      for (size_t i = 0; i < num_of_nodes; i++) {
        std::vector<double> curr_node = weights[i];
        double node_dot_product = std::inner_product(curr_node.begin(), curr_node.end(), prev_layer.begin(), 0.0) + biases[i];
        curr_layer[i] = activation_function(node_dot_product);
      }

      return curr_layer;
    }
};
