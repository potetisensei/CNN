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
    PoolLayer(int breadth_neuron, int num_channels, int stride, int breadth_filter);
    virtual ~PoolLayer() {}
    virtual void ConnectLayer(Layer *layer);
    virtual void CalculateOutput(Layer *layer);
    virtual void Propagate(Layer *layer);
    virtual void BackPropagate(DoubleVector2d next_deltas) { assert(0); }
    virtual void UpdateWeight(DoubleVector2d deltas) { assert(0); }
    virtual void UpdateBias(DoubleVector2d deltas) { assert(0); }

    bool calculated_;

    DoubleVector2d deltas_; // [sample_idx][neuron_idx] 
private:
    int breadth_neuron_;
    int num_channels_;
    int stride_;
    int breadth_filter_;
    int breadth_output_;

    vector<int> maxid;
};

#endif
