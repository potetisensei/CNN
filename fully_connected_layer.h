#ifndef _FULLY_CONNECTED_LAYER_H_
#define _FULLY_CONNECTED_LAYER_H_

using namespace std;

#include <vector>
#include "layer.h"
#include "util.h"

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(int num_neurons, ActivationFunction *f, double learning_rate);
    virtual ~FullyConnectedLayer() {}   
    virtual void ConnectLayer(Layer *layer);
    virtual void Propagate(Layer *layer);
    virtual void BackPropagate(DoubleVector2d next_deltas);
    virtual void UpdateWeight(DoubleVector2d deltas);
    virtual void UpdateBias(DoubleVector2d deltas) ;

private:
    EdgeVector2d edges_;
    vector<double> biases_;
};
   
#endif
