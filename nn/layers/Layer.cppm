module;

#include <vector>

export module nn.layers.Layer;

export class Layer {
public:
  Layer();
  Layer(Layer &&) = default;
  Layer(const Layer &) = default;
  Layer &operator=(Layer &&) = default;
  Layer &operator=(const Layer &) = default;
  ~Layer();
 
  virtual std::vector<double> calculate_layer(const std::vector<double>& prev_layer) = 0;

private:
  
};
