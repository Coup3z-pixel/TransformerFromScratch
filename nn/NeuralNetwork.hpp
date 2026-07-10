#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

namespace NN {
  class Network
  {
    public:
      Network();
      void train();
      void predict();

    private:
      void feed_forward();
      void back_prop();
  };
}



#endif // !NEURALNETWORK_H
