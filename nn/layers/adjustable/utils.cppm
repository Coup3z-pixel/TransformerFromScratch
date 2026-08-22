module;

#include <random>
#include <cmath>
#include <vector>

export module nn.layers.adjustable.utils;

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
