#ifndef _LAYER_H_
#define _LAYER_H_

#include <cassert>
#include <vector>
#include "activation_function.h"
#include "util.h"
using namespace std;

class Layer {
public:
    Layer() : calculated_(false), f_(&id_) {}
    Layer(double rate, ActivationFunction *f) 
        : calculated_(false), learning_rate_(rate), f_(f) {}
    virtual ~Layer() {}
    virtual void CheckInputUnits(vector<struct Neuron> const &units) { assert(0); } 
    virtual void ArrangeOutputUnits(vector<struct Neuron> &units) { assert(0); }
    virtual void ConnectNeurons(vector<struct Neuron> const &input, vector<struct Neuron> const &output) { assert(0); }
    virtual void CalculateOutputUnits(vector<struct Neuron> &output);
    virtual void Propagate(vector<struct Neuron> const &input, vector<struct Neuron> const &output) { assert(0); }
    virtual void BackPropagate(vector<struct Neuron> const &input, vector<struct Neuron> const &output) { assert(0); }
    virtual void UpdateWeights() { assert(0); }
/*
    bool calculated_;
    DoubleVector2d deltas_; // [sample_idx][neuron_idx] 
    ActivationFunction *f_;
public:
    double learning_rate_;
    Identity id_;*/
};

#endif
