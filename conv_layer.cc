#include "conv_layer.h"

ConvLayer::ConvLayer(
    int breadth_neuron, 
    int num_channels, 
    int stride, 
    int breadth_filter, 
    int num_filters, 
    ActivationFunction *f, 
    double learning_rate)
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

void ConvLayer::ConnectLayer(Layer *layer) {
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
void ConvLayer::Propagate(Layer *layer) {
    vector<struct Neuron> &output_neurons = layer->neurons_;
    int size = breadth_output_ * breadth_output_;
    int size2 = breadth_neuron_ * breadth_neuron_;

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
			    
                            int neuron_idx2 = k*size2 + y*breadth_neuron_ + x;
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

void BackPropagate( DoubleVector2d next_deltas ){
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
			    
                            int neuron_idx2 = k*size2 + y*breadth_neuron_ + x;
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
  
}
