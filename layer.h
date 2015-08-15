#ifndef _LAYER_H_
#define _LAYER_H_

#include <cassert>
#include <vector>
#include "activation_function.h"
#include "util.h"
using namespace std;

class Layer {
public:
    Layer() : calculated_(false) {}
    Layer(double rate, ActivationFunction *f) 
        : calculated_(false), learning_rate_(rate), f_(f) {}
    virtual ~Layer() {}
    virtual void ConnectLayer(Layer *layer) { assert(0); }
    virtual void CalculateOutput(Layer *layer);
    virtual void Propagate(Layer *layer) { assert(0); }
    virtual void BackPropagate(DoubleVector2d next_deltas) { assert(0); }
    virtual void UpdateWeight(DoubleVector2d deltas) { assert(0); }
    virtual void UpdateBias(DoubleVector2d deltas) { assert(0); }

    bool calculated_;
    vector<struct Neuron> neurons_; // think as 1d even if Layer has 2d or 3d neurons
    DoubleVector2d deltas_; // [sample_idx][neuron_idx] 
protected:
    ActivationFunction *f_;
    double learning_rate_;
};

#endif
