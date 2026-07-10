#include <iostream>
#include <cassert>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <functional>

enum class ActivationFunction {
  relu,
  sigmoid
};

std::vector<double> initialize_random_weights(int num_of_input_nodes) {
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

void softmax(std::vector<double> output_layer) {
  double exponent_sum = 0;

  for (double node : output_layer) {
    exponent_sum += std::exp(node);
  }

  for (int i = 0; i < output_layer.size(); i++) {
    output_layer[i] = std::exp(output_layer[i]) / exponent_sum;
  }
}

// Standard Sigmoid Activation Function: f(x) = 1 / (1 + e^(-x))
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Sigmoid Derivative: f'(x) = f(x) * (1 - f(x))
double sigmoid_derivative(double activated_val) {
    return activated_val * (1.0 - activated_val);
}

// ReLU Activation Function: f(x) = max(0, x)
double relu(double x) {
    return std::max(0.0, x);
}

// Leaky ReLU Derivative: f'(x) = 1 if f(x) > 0 else 0.01
double relu_derivative(double activated_val) {
    // Since activated_val = max(0, x), if it is greater than 0, 
    // the original net input 'x' must have been greater than 0.
    if (activated_val <= 0.0) {
        return 0.01; // Using your leaky fallback to prevent dead neurons
    }
    return 1.0;
}

class Node {
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
      bias = 0.01;
    }

    Node() = default;

    double process(std::vector<double> input) {
      double embedded_val = std::inner_product(weights.begin(), weights.end(), input.begin(), 0.0) + bias;

      switch (activation_func) {
        case ActivationFunction::relu:
          return std::max(0.0, embedded_val);
          break;
        case ActivationFunction::sigmoid:
          return embedded_val;
          break;
        default:
          break;
      }

      return embedded_val;
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

class HiddenLayer
{
  private:
    ActivationFunction activation_func_type;
    int num_of_nodes;
    std::vector<Node> nodes;

    std::function<double(double)> activation_function;
    std::function<double(double)> activation_derivative;

  public:
    HiddenLayer(
        int layer_node_count,
        ActivationFunction activ_func_type = ActivationFunction::relu
        ) {
      num_of_nodes = layer_node_count;

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

    Node& get_node(int index) {
      return nodes[index];
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
    
    void adjust_nodes_weight(int node_idx, int weight_idx, double adjust_factor) {
      nodes[node_idx].adjust_weight(weight_idx, adjust_factor);
    }

    void setup_layer(int prev_layer_node_num, int layer_idx = -1) {
      if (layer_idx != -1) {
        std::cout << "Init Layer " << layer_idx << " with " << prev_layer_node_num << " previous nodes \n";
      }

      nodes = std::vector<Node>(num_of_nodes);

      for (int n = 0; n < num_of_nodes; n++) {
        Node next_node = Node(prev_layer_node_num, activation_func_type);
        nodes[n] = next_node;
      }
    }

    std::vector<double> calculate_layer(std::vector<double> prev_layer) {
      std::vector<double> curr_layer = std::vector<double>(num_of_nodes);

      for (size_t i = 0; i < num_of_nodes; i++) {
        Node curr_node = nodes[i];
        curr_layer[i] = curr_node.process(prev_layer);
      }

      return curr_layer;
    }
};

class Network
{
  std::vector<HiddenLayer> layers;

  public:
  Network(int input_layer_count, std::vector<HiddenLayer> layer_config) {
    int prev_layer_node_num = input_layer_count;
    int layer_idx = 0;

    layers = std::vector<HiddenLayer>(layer_config.size());

    for (auto layer : layer_config) {
      HiddenLayer new_layer = HiddenLayer(layer.get_num_of_nodes(), layer.get_activation_func());
      new_layer.setup_layer(prev_layer_node_num);
      prev_layer_node_num = new_layer.get_num_of_nodes();
      layers[layer_idx] = new_layer;
      layer_idx++;
    }

    std::cout << "Size: " << layers.size() << "\n";
  }

  void train(std::vector<std::vector<double>> X, std::vector<std::vector<double>> Y, int epochs = 1000, double learning_rate = 0.003) {
    std::cout << "Training Has Started with " << epochs << " epochs \n";
    std::vector<std::vector<double>> activations = std::vector<std::vector<double>>(layers.size());
    std::vector<std::vector<double>> error_weights = std::vector<std::vector<double>>(layers.size());

    double loss = -1.0;
    double total_loss = 0;
    int dataset_size = X.size();

    for (int e = 0; e < epochs; e++) { // epochs
      total_loss = 0;
      for (int m = 0; m < X.size(); m++) { // training examples
        // feed forward
        std::vector<double> prev_layer = X[m];

        for (int l = 0; l < layers.size(); l++) {
          std::vector<double> layer_result = layers[l].calculate_layer(prev_layer);
          prev_layer = layer_result;
          activations[l] = layer_result;
          error_weights[l] = std::vector<double>(layer_result.size());
        }

        std::vector<double> true_output = Y[m];
        double loss_derivative = prev_layer[0] - true_output[0];
        loss = 0.5 * std::pow((true_output[0] - prev_layer[0]), 2); // 0.5(y - y')^2
        total_loss += loss;

        int num_of_nodes = layers[layers.size() - 1].get_num_of_nodes();

        for (int i = 0; i < activations[activations.size() - 1].size(); i++) {
          error_weights[error_weights.size() - 1][i] = loss_derivative * relu_derivative(activations[activations.size() - 1][i]);
        }

        // back propogation on the remaining layers
        for (int l = layers.size() - 2; l >= 0; l--) {
          int current_layer_nodes = layers[l].get_num_of_nodes();
          int next_layer_nodes = layers[l + 1].get_num_of_nodes();
          std::function<double(double)> layer_derivative_function = layers[l].get_activation_derivative();

          for (int n = 0; n < current_layer_nodes; n++) {
            double sum = 0.0;
            for (int next_n = 0; next_n < next_layer_nodes; next_n++) {
              double weight = layers[l + 1].get_node(next_n).get_weights()[n]; 
              sum += weight * error_weights[l + 1][next_n];
            }

            double node_activation = activations[l][n];
            error_weights[l][n] = layer_derivative_function(node_activation) * sum;
          }
        }

        for (int l = 0; l < layers.size(); l++) {
          for (int n = 0; n < layers[l].get_num_of_nodes(); n++) {
            double node_error = error_weights[l][n];

            double bias_gradient_step = learning_rate * node_error;
            layers[l].get_node(n).adjust_bias(bias_gradient_step);

            for (int w = 0; w < layers[l].get_node(n).get_num_of_weights(); w++) {
              double input_activation = (l == 0) ? X[m][w] : activations[l - 1][w];

              double gradient_step = learning_rate * node_error * input_activation;
              layers[l].adjust_nodes_weight(n, w, gradient_step);
            }
          }
        }
      }
      loss = total_loss / dataset_size;
      std::cout << "Epoch: " << (e + 1) << " | Loss: " << loss << "\n";
    }
  }

  std::vector<double> predict(const std::vector<double>& X) {
    return feed_forward(X);
  }

  private:
  std::vector<double> feed_forward(const std::vector<double>& X) {
    std::vector<double> prev_layer = X;

    for (HiddenLayer layer : layers) {
      std::vector<double> layer_result = layer.calculate_layer(prev_layer);
      prev_layer = layer_result;
    }

    return prev_layer;
  }
};

int main() {
  Network neural_network = Network(2, {
      HiddenLayer(3, ActivationFunction::relu),
      HiddenLayer(1, ActivationFunction::relu)
      });

  std::vector<std::vector<double>> X = {
    {0,0},
    {0,1},
    {1,0},
    {1,1}
  };

  std::vector<std::vector<double>> Y = {
    {0},
    {1},
    {1},
    {0}
  };

  neural_network.train(X, Y, 10000, 0.3);

  return 0;
}
