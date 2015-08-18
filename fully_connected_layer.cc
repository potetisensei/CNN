#include "fully_connected_layer.h"

FullyConnectedLayer::FullyConnectedLayer(int num_input, int num_output, ActivationFunction *f, double learning_rate) 
        : neuron_connected_(false),
          num_input_(num_input),
          num_output_(num_output),
          f_(f),
          learning_rate_(learning_rate) {}

void CheckInputUnits(vector<struct Neuron> const &units) {
    assert(units.size() == num_input_);
}

void ArrangeOutputUnits(vector<struct Neuron> &units) {
    units.resize(num_output_);
}

void FullyConnectedLayer::ConnectNeurons(
        vector<struct Neuron> const &input, 
        vector<struct Neuron> const &output) {
    assert(!neuron_connected_);
    assert(input.size() == num_input_);
    assert(output.size() == num_output_);
    
    biases_.resize(num_output_);
    for (int i=0; i<num_output_; i++) {
        struct Weight w;

        w.val = GenRandom(-0.5, 0.5)
        w.lazy_sub = 0.0;
        w.count = 0;
        biases_[i] = w;
    }

    weights_.resize(num_input_);
    for (int i=0; i<num_input_; i++) {
        weights_[i].resize(num_output_);
        for (int j=0; j<num_output_; j++) {
            struct Weight w;

            w.val = GenRandom(-1.0, 1.0);
            w.lazy_sub= 0.0;
            w.count = 0;
            weights_[i][j] = w;
        }
    }

    neuron_connected_ = true;
}

void FullyConnectedLayer::CalculateOutputUnits(vector<struct Neuron> &units) {
    assert(units.size() == num_output_);

    for (int i=0; i<num_output_; i++) {
        units[i].z = f_(units[i].u);
    }
}

void FullyConnectedLayer::Propagate(
        vector<struct Neuron> const &input, 
        vector<struct Neuron> &output) {
    assert(input.size() == num_input_);
    assert(output.size() == num_output_);
    
    for (int i=0; i<num_output_; i++) {
        output[i].u = 0.0;
    }

    for (int i=0; i<num_input_; i++) {
        for (int j=0; j<num_output_; j++) {
            struct Weight w = weights_[i][j];
            output[j].u += w.val * input[i].z;
        }
    }

    assert(biases_.size() == num_output_);
    for (int i=0; i<num_output_; i++) {
        output[i].u += biases_[i];
    }
}

void FullyConnectedLayer::BackPropagate(
        vector<struct Neuron> const &input,
        vector<double> const &next_delta,
        vector<double> &delta) {
    assert(input.size() == num_input_);
    assert(delta.size() == num_input_);
    assert(next_delta.size() == num_output_);

    assert(weights_.size() == num_input_);
    for (int i=0; i<num_input_; i++) {
        delta[i] = 0.0;

        assert(weights_[i].size() == num_output_);
        for (int j=0; j<num_output_; j++) {
            double w = weights_[i][j].val;

            delta[i] += 
                next_delta[j] * w * f_->CalculateDerivative(input[i].u);
        }
    }
}

void FullyConnectedLayer::UpdateLazySubtrahend(
        vector<struct Neuron> const &input,
        vector<double> const &next_delta) {
    assert(input.size() == num_input_);
    assert(next_delta.size() == num_output_);

    assert(weights_.size() == num_input_);
    for (int i=0; i<num_input_; i++) {
        assert(weights_[i].size() == num_output_);
        for (int j=0; j<num_output_; j++) {
            Weight &w = weights_[i][j];
            w.lazy_sub += learning_rate_ * next_delta[j] * input[i].z;
            w.count++;
        }
    }

    assert(biases_.size() == num_output_);
    for (int i=0; i<num_output_; i++) {
        Weight &w = biases_[i];
        w.lazy_sub += learning_rate_ * next_delta[i];
        w.count++;
    }
}

void FullyConnectedLayer::ApplyLazySubtrahend() {
    assert(weights_.size() == num_input_);
    for (int i=0; i<num_input_; i++) {
        assert(weights_[i].size() == num_output_);
        for (int j=0; j<num_output_; j++) {
            Weight &w = weights_[i][j];

            assert(w.count > 0);
            w.val -= w.lazy_sub / w.count;
            w.lazy_sub = 0.0;
            w.count = 0;
        }
    }

    assert(biases_.size() == num_output_);
    for (int i=0; i<num_output_; i++) {
        Weight &w = biases_[i];

        assert(w.count > 0);
        w.val -= w.lazy_sub / w.count;
        w.lazy_sub = 0.0;
        w.count = 0;
    }
}
