module;

#include <vector>
#include <iostream>
#include <functional>
#include <cmath>

export import loss;
export module nn.Network;
export import nn.Layer;

export class Network
{
  std::vector<HiddenLayer> layers;
  std::function<double(std::vector<double>, std::vector<double>)> loss_function;
  std::function<std::vector<double>(std::vector<double>, std::vector<double>)> loss_derivative_function;

  public:
  Network(
      int input_layer_count,
      std::vector<HiddenLayer> layer_config,
      LossFunctions loss_func_type = LossFunctions::MeanSquared
      ) {
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

    double loss = -1.0;
    double total_loss = 0;
    int correct_guesses = 0;
    int dataset_size = X_train.size();

    for (int e = 0; e < epochs; e++) {
      total_loss = 0;
      for (int m = 0; m < X_train.size(); m++) {
        std::vector<double> prev_layer = X_train[m];

        for (int l = 0; l < layers.size(); l++) {
          std::vector<double> layer_result = layers[l].calculate_layer(prev_layer);
          prev_layer = layer_result;
          activations[l] = layer_result;
          error_weights[l] = std::vector<double>(layer_result.size());
        }

        std::vector<double> true_output = Y_train[m];
        std::vector<double> loss_derivatives = loss_derivative_function(Y_train[m], prev_layer);
        loss = loss_function(Y_train[m], prev_layer);

        total_loss += loss;
        if (m % 10 == 0) {
          std::cout << m << ": " << total_loss / 10.0 << "\n";
          total_loss = 0;
        }

        double max_seen = -1;
        int model_guess = -1;

        for (auto i{0}; i < prev_layer.size(); i++) {
          if (max_seen < prev_layer[i]) {
            max_seen = prev_layer[i];
            model_guess = i;
          }
        }

        correct_guesses += Y_train[m][model_guess] == 1.0;

        if (m % 100 == 0 && prev_layer.size() > 1) {
          std::cout << "Accuracy: " << correct_guesses / 100.0 << "\n";
          correct_guesses = 0;
        }

        int num_of_nodes = layers[layers.size() - 1].get_num_of_nodes();

        std::function<double(double)> output_layer_activation_derivative = layers[layers.size() - 1].get_activation_derivative();

        for (int i = 0; i < activations[activations.size() - 1].size(); i++) {
          error_weights[error_weights.size() - 1][i] = loss_derivatives[i] * output_layer_activation_derivative(activations[activations.size() - 1][i]);
        }

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
              double input_activation = (l == 0) ? X_train[m][w] : activations[l - 1][w];

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
