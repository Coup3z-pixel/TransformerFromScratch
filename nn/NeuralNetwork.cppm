module;

#include <vector>
#include <iostream>
#include <functional>
#include <cmath>
#define PRINT_EVERY 100

export import loss;
export module nn.Network;
import nn.layers.adjustable.HiddenLayer;

export class Network
{
  std::vector<HiddenLayer> layers;
  std::function<double(std::vector<double>, std::vector<double>)> loss_function;
  // derivative with respect to activation (not weighted sum)
  std::function<std::vector<double>(std::vector<double>, std::vector<double>)> loss_derivative_function;
  LossFunctions loss_function_type;

  long total_loss;

  public:
  Network(
      int input_layer_count,
      std::vector<HiddenLayer> layer_config,
      LossFunctions loss_func_type = LossFunctions::MeanSquared
      ) {
    loss_function_type = loss_func_type;
    int prev_layer_node_num = input_layer_count;
    int layer_idx = 0;

    layers = std::vector<HiddenLayer>(layer_config.size());

    for (auto& layer : layer_config) {
      HiddenLayer new_layer = HiddenLayer(layer.get_num_of_nodes(), layer.get_activation_func());
      new_layer.setup_layer(prev_layer_node_num);
      prev_layer_node_num = new_layer.get_num_of_nodes();
      layers[layer_idx] = new_layer;
      layer_idx++;
    }

    switch (loss_func_type) {
      case LossFunctions::MeanSquared:
        loss_function = mse_loss;
        loss_derivative_function = mse_deriv_loss;
        break;
      case LossFunctions::CatergoricalCross:
        loss_function = cce_loss;
        loss_derivative_function = cce_deriv_loss;
        break;
      default:
        break;
    }
  }

  void train(
      const std::vector<std::vector<double>>& X_train,
      const std::vector<std::vector<double>>& Y_train,
      const std::vector<std::vector<double>>& X_test,
      const std::vector<std::vector<double>>& Y_test,
      int epochs = 1000,
      double learning_rate = 0.003
    ) {
    std::cout << "Training Has Started with " << epochs << " epochs \n";
    std::vector<std::vector<double>> activations = std::vector<std::vector<double>>(layers.size());
    std::vector<std::vector<double>> error_weights = std::vector<std::vector<double>>(layers.size());

    total_loss = 0;
    int correct_guesses = 0;
    int dataset_size = X_train.size();

    for (int l{}; l < layers.size(); l++) {
      error_weights[l] = std::vector<double>(layers[l].get_num_of_nodes());
    }

    for (int e{}; e < epochs; e++) {
      total_loss = 0;
      for (int m{}; m < X_train.size(); m++) {
        // setting up the input layer
        std::vector<double> prev_layer = X_train[m];

        // feed forward
        for (int l{}; l < layers.size(); l++) {
          std::vector<double> layer_result = layers[l].calculate_layer(prev_layer);
          prev_layer = layer_result;
          activations[l] = layer_result;
        }

        std::vector<double> true_output = Y_train[m];
        std::vector<double> loss_derivatives = loss_derivative_function(prev_layer, true_output);
        total_loss += loss_function(prev_layer, Y_train[m]);;

        training_telemetry(true_output, prev_layer, m);

        int num_of_nodes = layers[layers.size() - 1].get_num_of_nodes();

        std::function<double(double)> output_layer_activation_derivative = layers[layers.size() - 1].get_activation_derivative();

        int activation_final_layer_idx = activations.size() - 1;
        int error_weights_final_layer_idx = error_weights.size() - 1;

        // Getting the partial derivative of the loss function
        // With respect to the activation of the output layer nodes
        for (int i{}; i < activations[activation_final_layer_idx].size(); i++) {
          double output_layer_derivative = output_layer_activation_derivative(activations[activation_final_layer_idx][i]);
          double activation_layer_derivative = loss_derivatives[i] * output_layer_derivative;
          error_weights[error_weights_final_layer_idx][i] = activation_layer_derivative;
        }

        // carry the delta backwards
        for (int l = layers.size() - 2; l >= 0; l--) {
          int current_layer_nodes = layers[l].get_num_of_nodes();
          int next_layer_nodes = layers[l + 1].get_num_of_nodes();
          std::function<double(double)> layer_derivative_function = layers[l].get_activation_derivative();

          auto weights = layers[l + 1].get_layer_weights();

          // Initialize sum array
          std::vector<double> sum(current_layer_nodes, 0.0);

          // Inner loop over next_layer_nodes for better cache hit rate
          for (int next_n{}; next_n < next_layer_nodes; next_n++) {
            double ew = error_weights[l + 1][next_n];
            const auto& next_weights = weights[next_n];
            for (int n{}; n < current_layer_nodes; n++) {
              sum[n] += next_weights[n] * ew;
            }
          }

          for (int n{}; n < current_layer_nodes; n++) {
            double node_activation = activations[l][n];
            error_weights[l][n] = layer_derivative_function(node_activation) * sum[n];
          }
        }

        // update weights
        for (int l{}; l < layers.size(); l++) {
          for (int n{}; n < layers[l].get_num_of_nodes(); n++) {
            double node_error = error_weights[l][n];

            double bias_gradient_step = learning_rate * node_error;
            layers[l].adjust_bias(n, bias_gradient_step);

            for (int w{}; w < layers[l].get_num_of_weights(); w++) {
              double input_activation = (l == 0) ? X_train[m][w] : activations[l - 1][w];

              double gradient_step = learning_rate * node_error * input_activation;
              layers[l].adjust_nodes_weight(n, w, gradient_step);
            }
          }
        }
      }
      std::cout << "Epoch: " << (e + 1) << " | Loss: " << (total_loss / dataset_size) << "\n";
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

  void training_telemetry(const std::vector<double>& true_y, const std::vector<double>& pred_y, int m) {
    if (m == 0) return;

    if (m % PRINT_EVERY == 0 && pred_y.size() > 1) {
      std::cout << m << ": " << total_loss / static_cast<float>(m) << "\n";
      total_loss = 0;
    }
  }
};
