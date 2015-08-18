#ifndef _LAYER_H_
#define _LAYER_H_

#include <cassert>
#include <vector>
#include "activation_function.h"
#include "util.h"
using namespace std;

class Layer {
public:
    Layer() {}
    virtual ~Layer() {}
    virtual void CheckInputUnits(vector<struct Neuron> const &units);
    virtual void ArrangeOutputUnits(vector<struct Neuron> &units);
    virtual void ConnectNeurons(vector<struct Neuron> const &input, vector<struct Neuron> const &output);
    virtual void CalculateOutputUnits(vector<struct Neuron> &units);
    virtual void Propagate(vector<struct Neuron> const &input, vector<struct Neuron> &output);
    virtual void BackPropagate(vector<struct Neuron> const &input, vector<double> const &next_delta, vector<double> &delta);
    virtual void UpdateLazySubtrahend(vector<struct Neuron> const &input, const vector<double> &next_delta);
    virtual void ApplyLazySubtrahend();
};

#endif
