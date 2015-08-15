#include "neural_net.h"

void NeuralNet::GetOutput(vector<double> &output) {
    int last_idx = layers_.size()-1;

    assert(last_idx >= 1);
    assert(layers_[last_idx]->calculated_);
    vector<struct Neuron> &neurons = layers_[last_idx]->neurons_;
    assert(neurons.size() == output.size());

    for (int i=0; i<neurons.size(); i++) {
        output[i] = neurons[i].z;
    }
}

void NeuralNet::AppendLayer(Layer *layer) {
    assert(!connected_);
    layers_.push_back(layer);
}

void NeuralNet::ConnectLayers() {
    assert(!connected_);
    for (int i=0; i<layers_.size()-1; i++) {
        layers_[i]->ConnectLayer(layers_[i+1]);
    }
    connected_ = true;
}

void NeuralNet::PropagateLayers(vector<double> &input, vector<double> &output) {
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

void NeuralNet::BackPropagateLayers(DoubleVector2d &dataset, DoubleVector2d &outputs) {
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

void NeuralNet::TrainNetwork(DoubleVector2d &inputs, DoubleVector2d &expected_outputs) {
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
