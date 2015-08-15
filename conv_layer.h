#ifndef _CONV_LAYER_H_
#define _CONV_LAYER_H_

#include <cassert>
#include <vector>
#include "layer.h"
#include "util.h"
using namespace std;

class ConvLayer : public Layer {
public:
    ConvLayer(
        int breadth_neuron, 
        int num_channels, 
        int stride, 
        int breadth_filter, 
        int num_filters, 
        ActivationFunction *f, 
        double learning_rate);

    virtual ~ConvLayer() {}
    virtual void ConnectLayer(Layer *layer);
    virtual void Propagate(Layer *layer);
    virtual void BackPropagate(DoubleVector2d next_deltas);
    virtual void UpdateWeight(DoubleVector2d deltas) { }
    virtual void UpdateBias(DoubleVector2d deltas) { }

private:
    int breadth_neuron_;
    int num_channels_;
    int stride_;
    int breadth_filter_;
    int num_filters_;
    int breadth_output_;
    vector<double> biases_;
    DoubleVector4d edges_weight_; // [filter_idx][channel_idx][y][x] weight-sharing
};

#endif
