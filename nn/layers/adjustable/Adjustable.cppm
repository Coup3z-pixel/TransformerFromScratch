module;

#include <vector>
#include <iostream>
#include <functional>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

export module nn.layers.adjustable.Adjustable;
import nn.ActivationFunction;

export class AdjustableLayer {
public:
  virtual void setup_layer(int prev_layer_node_num, int layer_idx = -1) = 0;
  virtual int get_num_of_nodes() = 0;
  virtual int get_num_of_weights() = 0;
  virtual std::vector<std::vector<double>> get_layer_weights() = 0;
  virtual ActivationFunction get_activation_func() = 0;
  virtual std::function<double(double)> get_activation_function() = 0;
  virtual std::function<double(double)> get_activation_derivative() = 0;
  virtual void adjust_bias(int node_idx, double adjust_factor) = 0;
  virtual void adjust_nodes_weight(int node_idx, int weight_idx, double adjust_factor) = 0;

private:
  
};
