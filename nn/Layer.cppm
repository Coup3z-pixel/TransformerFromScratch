module;

#include <vector>
#include <iostream>
#include <functional>

export module nn.Layer;
export import nn.Node;

export class HiddenLayer
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
        Node& curr_node = nodes[i];
        curr_layer[i] = activation_function(curr_node.process(prev_layer));
      }

      return curr_layer;
    }
};
