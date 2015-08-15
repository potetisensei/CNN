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
    NeuralNet() : connected_(false)  {}
    ~NeuralNet() {}

    void GetOutput(vector<double> &output);
    void AppendLayer(Layer *layer);
    void ConnectLayers();
    void PropagateLayers(vector<double> &input, vector<double> &output);
    void BackPropagateLayers(DoubleVector2d &dataset, DoubleVector2d &outputs);
    void TrainNetwork(DoubleVector2d &inputs, DoubleVector2d &expected_outputs);

private:
    bool connected_;
    vector<Layer*> layers_;
};

#endif
