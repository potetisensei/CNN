#ifndef _FULLY_CONNECTED_LAYER_H_
#define _FULLY_CONNECTED_LAYER_H_

using namespace std;

#include <vector>
#include "layer.h"
#include "activation_function.h"
#include "util.h"

class FullyConnectedLayer : public Layer {
public:
  FullyConnectedLayer(int num_input, int num_output, ActivationFunction *f, double learning_rate, double momentum, double dropout_rate);
    virtual ~FullyConnectedLayer() {}   

    virtual void CheckInputUnits(vector<struct Neuron> const &units);
    virtual void ArrangeOutputUnits(vector<struct Neuron> &units);
    virtual void ConnectNeurons(vector<struct Neuron> const &input, vector<struct Neuron> const &output);
    virtual void CalculateOutputUnits(vector<struct Neuron> &units);
    virtual void Propagate(vector<struct Neuron> const &input, vector<struct Neuron> &output);
    virtual void BackPropagate(vector<struct Neuron> const &input, vector<double> const &next_delta, ActivationFunction *f, vector<double> &delta);
    virtual void UpdateLazySubtrahend(vector<struct Neuron> const &input, const vector<double> &next_delta);
    virtual void ApplyLazySubtrahend();

    virtual void Save( char *s );
    virtual void Load( char *s );
    
private:
    bool neuron_connected_;
    int num_input_;
    int num_output_;
    double learning_rate_;
    double momentum_;
    WeightVector2d weights_;
    vector<struct Weight> biases_;
};
   
#endif
