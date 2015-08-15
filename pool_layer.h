#ifndef _POOL_LAYER_H_
#define _POOL_LAYER_H_

#include <vector>
#include "layer.h"
#include "util.h"
using namespace std;

class PoolLayer : public Layer {
    PoolLayer(int breadth_neuron, int num_channels, int stride, int breadth_filter)
        : breadth_neuron_(breadth_neuron), 
          stride_(stride),
          breadth_filter_(breadth_filter) {
        breadth_output_ = (breadth_neuron-1)/stride + 1;
    }

    virtual ~PoolLayer() {}

    virtual void ConnectLayer(Layer *layer) { assert(0); }

    virtual void CalculateOutput(Layer *layer) {
        vector<struct Neuron> &neurons = layer->neurons_;

        assert(!layer->calculated_);
        for (int i=0; i<neurons.size(); i++) {
            neurons[i].z = f_->Calculate(neurons[i].u, neurons);
        }
        layer->calculated_ = true;
    }

    virtual void Propagate(Layer *layer) { assert(0); }
    virtual void BackPropagate(DoubleVector2d next_deltas) { assert(0); }
    virtual void UpdateWeight(DoubleVector2d deltas) { assert(0); }
    virtual void UpdateBias(DoubleVector2d deltas) { assert(0); }

    bool calculated_;
    vector<struct Neuron> neurons_; // think as 1d even if Layer has 2d or 3d neurons
    DoubleVector2d deltas_; // [sample_idx][neuron_idx] 
private:
    int breadth_neuron_;
    int num_channels_;
    int stride_;
    int breadth_filter_;
    int breadth_output_;
};

#endif
