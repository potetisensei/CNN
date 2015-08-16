#include "pool_layer.h"
#include "iostream"

PoolLayer::PoolLayer(int breadth_neuron, int num_channels, int stride, int breadth_filter)
    : breadth_neuron_(breadth_neuron), 
      stride_(stride),
      breadth_filter_(breadth_filter),
      num_channels_(num_channels){
    breadth_output_ = (breadth_neuron-1)/stride + 1;
    neurons_.resize( breadth_neuron * breadth_neuron * num_channels );
    maxid.resize(breadth_output_ * breadth_output_ * num_channels_);
}

void PoolLayer::CalculateOutput(Layer *layer) {
    vector<struct Neuron> &neurons = layer->neurons_;

    assert(!layer->calculated_);
    for (int i=0; i<neurons.size(); i++) {
      neurons[i].z = neurons[i].u;
    }
    layer->calculated_ = true;
}

void PoolLayer::ConnectLayer(Layer *layer){
  assert( breadth_output_*breadth_output_*num_channels_ == layer->neurons_.size() );
}

void PoolLayer::Propagate(Layer *layer){
  vector<struct Neuron> &output_neurons = layer->neurons_;
  int size = breadth_output_ * breadth_output_;
  int size2 = breadth_neuron_ * breadth_neuron_;

  assert(calculated_);

  for (int i=0; i<output_neurons.size(); i++) {
    output_neurons[i].u = 0.0;
  }
  
  assert(maxid.size() == num_channels_*size);
  for (int i=0; i<breadth_output_; i++) {
    for (int j=0; j<breadth_output_; j++) {
      for (int k=0; k<num_channels_; k++) {
        int neuron_idx = k*size + i*breadth_output_ + j;
	    double maxv = -numeric_limits<double>::max();

        assert(j*stride_ < breadth_neuron_);
        assert(i*stride_ < breadth_neuron_);

    	for (int p=0; p<breadth_filter_; p++) {
	      for (int q=0; q<breadth_filter_; q++) {
    	    int x = j*stride_+q;
	        int y = i*stride_+p;
	        double z = -numeric_limits<double>::max();
            int neuron_idx2 = k*size2 + y*breadth_neuron_ + x;

            printf("x, y: %d %d\n", x, y);
            printf("breadth_neuron_: %d\n", breadth_neuron_);
	        if (x < breadth_neuron_ && y < breadth_neuron_) {
	          z = neurons_[neuron_idx2].z;
              printf("z: %f\n", z);
            }

    	    if (maxv < z) {
	          maxv = z;
	          maxid[neuron_idx] = neuron_idx2;
	        }
          }
	    }

   	    assert(maxv != -numeric_limits<double>::max());
	    output_neurons[neuron_idx].u = maxv;
	  }
    }
  }

  layer->calculated_ = false;
}


void PoolLayer::BackPropagate(DoubleVector2d next_deltas) {
  deltas_.resize(next_deltas.size());
  for (int i=0; i<next_deltas.size(); i++) {
    deltas_[i].resize(neurons_.size());
    fill(deltas_[i].begin(), deltas_[i].end(), 0.0);

    for (int j=0; j<next_deltas[i].size(); j++) {
      deltas_[i][maxid[j]] += next_deltas[i][j];
    }
  }
}
