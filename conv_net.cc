#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>
#include <vector>
#include <limits>
#include "bmp.h"
#include "util.h"
#include "layer.h"
#include "fully_connected_layer.h"

using namespace std;

class ConvLayer : public Layer {
public:
    ConvLayer(int breadth_neuron, int num_channels, int stride, int breadth_filter, int num_filters, ActivationFunction *f, double learning_rate)
            :  breadth_neuron_(breadth_neuron), 
               num_channels_(num_channels), 
               stride_(stride), 
               breadth_filter_(breadth_filter),
               num_filters_(num_filters),
               Layer(learning_rate, f) {
        assert(stride >= 1);
        neurons_.resize(breadth_neuron * breadth_neuron * num_channels); // dangerous
        edges_weight_.resize(num_filters);
        breadth_output_ = (breadth_neuron-1)/stride + 1;
    }

    virtual ~ConvLayer() {}

    virtual void ConnectLayer(Layer *layer) {
        int expected_size = num_filters_ * breadth_output_ * breadth_output_; // dangerous

        assert(layer->neurons_.size() == expected_size);

        for (int i=0; i<num_filters_; i++) { // Bijm == Bm
            biases_.push_back(GenRandom(-0.5, 0.5));
        }

        double lim = 1.0/sqrt(breadth_filter_*breadth_filter_*num_channels_);
        assert(edges_weight_.size() == num_filters_);
        for (int m=0; m<num_filters_; m++) {
            edges_weight_[m].resize(num_channels_);
            for (int k=0; k<num_channels_; k++) {
                edges_weight_[m][k].resize(breadth_filter_);
                for (int i=0; i<breadth_filter_; i++) {
                    edges_weight_[m][k][i].resize(breadth_filter_);
                    for (int j=0; j<breadth_filter_; j++) {
                        edges_weight_[m][k][i][j] = GenRandom(-lim, lim);
                    }
                }
            }
        }
    }

 
    //    3d_neurons[z][y][x] z := m, k, z...
    //                        y := i, p, y...
    //                        x := j, q, x...
    virtual void Propagate(Layer *layer) {
        vector<struct Neuron> &output_neurons = layer->neurons_;
        int size = breadth_output_ * breadth_output_;

        assert(calculated_);

        for (int i=0; i<output_neurons.size(); i++) {
            output_neurons[i].u = 0.0;
        }

        assert(biases_.size() == num_filters_);
        for (int m=0; m<num_filters_; m++) { 
            for (int i=0; i<breadth_output_; i++) {
                for (int j=0; j<breadth_output_; j++) {
                    int neuron_idx1 = m*size + i*breadth_output_ + j;
                    double sum_conv = 0.0;

                    assert(i*stride_ < breadth_neuron_);
                    assert(j*stride_ < breadth_neuron_);

                    for (int k=0; k<num_channels_; k++) {
                        for (int p=0; p<breadth_filter_; p++) {
                            for (int q=0; q<breadth_filter_; q++) {
                                int x = j*stride_ + q;
                                int y = i*stride_ + p;
                                int neuron_idx2 = k*size + y*breadth_output_ + x;
                                double z = 0.0;

                                if (x < breadth_neuron_ && y < breadth_neuron_) {
                                    z = neurons_[neuron_idx2].z;
                                }
                                sum_conv += z * edges_weight_[m][k][p][q];
                            }
                        }
                    }

                    output_neurons[neuron_idx1].u = sum_conv + biases_[m];
                }
            }
        }

        layer->calculated_ = false;
    }

    virtual void BackPropagate(DoubleVector2d next_deltas) { assert(0); }
    virtual void UpdateWeight(DoubleVector2d deltas) { assert(0); }
    virtual void UpdateBias(DoubleVector2d deltas) { assert(0); }

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

class ConvNet {
public:
    ConvNet(/*ErrorFunction *e*/) : connected_(false)/*, e_(e)*/  {}
    ~ConvNet() {}

    void GetOutput(vector<double> &output) {
        int last_idx = layers_.size()-1;

        assert(last_idx >= 1);
        assert(layers_[last_idx]->calculated_);
        vector<struct Neuron> &neurons = layers_[last_idx]->neurons_;
        assert(neurons.size() == output.size());

        for (int i=0; i<neurons.size(); i++) {
            output[i] = neurons[i].z;
        }
    }

    void AppendLayer(Layer *layer) {
        assert(!connected_);
        layers_.push_back(layer);
    }

    void ConnectLayers() {
        assert(!connected_);   
        for (int i=0; i<layers_.size()-1; i++) {
            layers_[i]->ConnectLayer(layers_[i+1]);
        }
        connected_ = true;
    }

    void PropagateLayers(vector<double> &input, vector<double> &output) {
        int last_idx = layers_.size()-1;

        assert(connected_);
        assert(last_idx >= 1);

        vector<struct Neuron> &first_neurons = layers_[0]->neurons_;
        vector<struct Neuron> &last_neurons = layers_[last_idx]->neurons_;
        assert(first_neurons.size() == input.size());
        assert(last_neurons.size() == output.size());

        for (int i=0; i<first_neurons.size(); i++) {
            first_neurons[i].z = input[i];
        }

        layers_[0]->calculated_ = true;
        for (int i=0; i<=last_idx-1; i++) {
            layers_[i]->Propagate(layers_[i+1]);
            layers_[i]->CalculateOutput(layers_[i+1]);
        }

        assert(layers_[last_idx]->calculated_);
        for (int i=0; i<last_neurons.size(); i++) {
            output[i] = last_neurons[i].z;
        }
    }

    void BackPropagateLayers(DoubleVector2d &dataset, DoubleVector2d &outputs) {
        int last_idx = layers_.size()-1;

        assert(last_idx >= 1);
        assert(dataset.size() == outputs.size());
        DoubleVector2d &deltas = layers_[last_idx]->deltas_;

        deltas.resize(dataset.size());
        for (int i=0; i<dataset.size(); i++) {
            assert(dataset[i].size() == outputs[i].size());
            deltas[i].resize(dataset[i].size());
            for (int j=0; j<dataset[i].size(); j++) {
                deltas[i][j] = outputs[i][j] - dataset[i][j];
            }
        }

        layers_[last_idx-1]->UpdateWeight(deltas);
        layers_[last_idx-1]->UpdateBias(deltas);
        for (int i=last_idx-1; i>=1; i--) {
             layers_[i]->BackPropagate(layers_[i+1]->deltas_);
             layers_[i-1]->UpdateWeight(layers_[i]->deltas_);
             layers_[i-1]->UpdateBias(layers_[i]->deltas_);
        }
    }
    
    void TrainNetwork(DoubleVector2d &inputs, DoubleVector2d &expected_outputs) {
        DoubleVector2d actual_outputs;

        assert(inputs.size() == expected_outputs.size());
        actual_outputs.resize(inputs.size());
        for (int i=0; i<inputs.size(); i++) {
            assert(inputs[i].size() == expected_outputs[i].size());
            actual_outputs[i].resize(inputs[i].size());
            PropagateLayers(inputs[i], actual_outputs[i]);
        }

        BackPropagateLayers(expected_outputs, actual_outputs);
    }

private:
    bool connected_;
    vector<Layer *> layers_;
    //ErrorFunction *e_;
};
