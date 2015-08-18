#include "fully_connected_layer.h"

FullyConnectedLayer::FullyConnectedLayer(int num_neurons, ActivationFunction *f, double learning_rate) 
        : Layer(learning_rate, f) {
    neurons_.resize(num_neurons);
    edges_.resize(num_neurons);
}

void FullyConnectedLayer::ConnectLayer(Layer *layer) {
    int num_output_neurons = layer->neurons_.size();
    
    for (int i=0; i<num_output_neurons; i++) {
        biases_.push_back(GenRandom(-0.5, 0.5));
    }

    for (int i=0; i<neurons_.size(); i++) {
        edges_[i].resize(num_output_neurons);
        for (int j=0; j<num_output_neurons; j++) {
            edges_[i][j].to = j;
            edges_[i][j].w = GenRandom(-1.0, 1.0);
        }
    }
}

void FullyConnectedLayer::Propagate(Layer *layer) {
    assert(calculated_);

    for (int i=0; i<layer->neurons_.size(); i++) {
        layer->neurons_[i].u = 0.0;
    }

    for (int i=0; i<neurons_.size(); i++) {
        for (int j=0; j<edges_[i].size(); j++) {
            struct Edge e = edges_[i][j];
            layer->neurons_[e.to].u += e.w * neurons_[i].z;
        }
    }

    assert(biases_.size() == layer->neurons_.size());
    for (int i=0; i<biases_.size(); i++) {
        layer->neurons_[i].u += biases_[i];
    }

    layer->calculated_ = false;
}

void FullyConnectedLayer::BackPropagate(DoubleVector2d next_deltas, ActivationFunction *f) {
  double deltamax = -1000;
  double deltamin = 1000;
    deltas_.resize(next_deltas.size());
    for (int i=0; i<deltas_.size(); i++) {
        vector<double> &next_delta = next_deltas[i];
        vector<double> &delta = deltas_[i];
        
        delta.resize(neurons_.size());
        fill(delta.begin(), delta.end(), 0.0);

        for (int j=0; j<edges_.size(); j++) {
            for (int i=0; i<edges_[j].size(); i++) {
                int k = edges_[j][i].to;
                double w = edges_[j][i].w;

                delta[j] += next_delta[k] * w * f->CalculateDerivative(neurons_[j].u);
		deltamax = max( deltamax , delta[j] );
		deltamin = min( deltamin , delta[j] );
            }
        }
    }
    //printf( "fullydelta %lf %lf\n" , deltamax , deltamin );
}

void FullyConnectedLayer::UpdateWeight(DoubleVector2d deltas) {
    for (int l=0; l<deltas.size(); l++) {
        assert(neurons_.size() == edges_.size());
        for (int i=0; i<edges_.size(); i++) {
            for (int k=0; k<edges_[i].size(); k++) {
                int j = edges_[i][k].to;
    
                edges_[i][k].w -= learning_rate_ * deltas[l][j] * neurons_[i].z / deltas.size(); // Wji -= e * delta[j] * z[i] / N
            }
        }
    }
}

void FullyConnectedLayer::UpdateBias(DoubleVector2d deltas) {
    for (int l=0; l<deltas.size(); l++) {
        assert(deltas[l].size() == biases_.size());
        for (int i=0; i<deltas[l].size(); i++) {
            biases_[i] -= learning_rate_ * deltas[l][i] / deltas.size(); // bi -= e * delta[i] / N
        }
    }
}
