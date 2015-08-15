#include "pool_layer.h"

PoolLayer::PoolLayer(int breadth_neuron, int num_channels, int stride, int breadth_filter)
    : breadth_neuron_(breadth_neuron), 
      stride_(stride),
      breadth_filter_(breadth_filter) {
    breadth_output_ = (breadth_neuron-1)/stride + 1;
}

void PoolLayer::CalculateOutput(Layer *layer) {
    vector<struct Neuron> &neurons = layer->neurons_;

    assert(!layer->calculated_);
    for (int i=0; i<neurons.size(); i++) {
        neurons[i].z = f_->Calculate(neurons[i].u, neurons);
    }
    layer->calculated_ = true;
}
