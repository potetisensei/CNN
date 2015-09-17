#ifndef _POOL_LAYER_H_
#define _POOL_LAYER_H_

#include <vector>
#include <algorithm>
#include <limits>
#include "layer.h"
#include "util.h"
using namespace std;

class PoolLayer : public Layer {
 public:
    PoolLayer(
        int breadth_neuron, 
        int num_channels, 
        int stride, 
        int breadth_filter ,
	ActivationFunction *f,
	double dropout_rate);
    virtual ~PoolLayer() {}

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

    void CalculateStyleMatrix(vector<struct Neuron> &units);
 
    
private:
    bool neuron_connected_;
    bool propagated_;
    int breadth_neuron_;
    int num_channels_;
    int stride_;
    int breadth_filter_;
    int breadth_output_;
    int num_input_;
    int num_output_;
    vector<int> maxid;
};

#endif
