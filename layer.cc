#include "layer.h"

void Layer::CalculateOutput(Layer *layer) {
    vector<struct Neuron> &neurons = layer->neurons_;

    assert(!layer->calculated_);
    for (int i=0; i<neurons.size(); i++) {
        neurons[i].z = f_->Calculate(neurons[i].u, neurons);
    }
    layer->calculated_ = true;
}
