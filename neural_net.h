#ifndef _NEURAL_NET_H_
#define _NEURAL_NET_H_

#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include "util.h"
#include "layer.h"
using namespace std;

class NeuralNet {
public:
    NeuralNet();
    ~NeuralNet() {}

    void SetInputSize(int size);
    void AppendLayer(Layer *layer);
    void ConnectLayers();
    void PropagateLayers(vector<double> &input, vector<double> &output, bool is_learning = false );
    void BackPropagateLayers(vector<double> &expected);
    void TrainNetwork(DoubleVector2d &inputs, DoubleVector2d &expected_outputs);

    void Save( string s );
    void Load( string s );

    void Visualize( int filenum , int depth , int size , int channel_n );

    void SetLearningFlag( int layer_n , bool f );
private:
    bool layer_connected_;
    NeuronVector2d all_neurons_;
    vector<Layer*> layers_;

    vector<bool> learning_f_;
};

#endif
