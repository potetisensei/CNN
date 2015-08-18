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

    virtual void CheckInputUnits(vector<struct Neuron> const &units);
    virtual void ArrangeOutputUnits(vector<struct Neuron> &units);
    virtual void ConnectNeurons(vector<struct Neuron> const &input, vector<struct Neuron> const &output);
    virtual void CalculateOutputUnits(vector<struct Neuron> &units);
    virtual void Propagate(vector<struct Neuron> const &input, vector<struct Neuron> &output);
    virtual void BackPropagate(vector<struct Neuron> const &input, vector<double> const &next_delta, ActivationFunction *f, vector<double> &delta);
    virtual void UpdateLazySubtrahend(vector<struct Neuron> const &input, const vector<double> &next_delta);
    virtual void ApplyLazySubtrahend();

private:
    bool neuron_connected_;
    int breadth_neuron_;
    int num_channels_;
    int stride_;
    int breadth_filter_;
    int num_filters_;
    int breadth_output_;
    int num_input_;
    int num_output_;
    double learning_rate_;
    vector<struct Weight> biases_;
    WeightVector4d weights_; // [filter_idx][channel_idx][y][x], weight-sharing
};

#endif
