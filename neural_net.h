#ifndef _NEURAL_NET_H_
#define _NEURAL_NET_H_

#include <cstdio>
#include <cassert>
#include <vector>
#include "util.h"
#include "layer.h"
using namespace std;

class NeuralNet {
public:
    NeuralNet() : layer_connected_(false)  {}
    ~NeuralNet() {}

    void AppendLayer(Layer *layer);
    void ConnectLayers();
    void PropagateLayers(vector<double> &input, vector<double> &output);
    void BackPropagateLayers(vector<double> &input, vector<double> &output);
    void TrainNetwork(DoubleVector2d &inputs, DoubleVector2d &expected_outputs);

private:
    bool layer_connected_;
    vector<Layer*> layers_;
};

#endif
