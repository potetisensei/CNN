#include "pool_layer.h"
#include "iostream"

PoolLayer::PoolLayer(int breadth_neuron, int num_channels, int stride, int breadth_filter)
    : breadth_neuron_(breadth_neuron), 
      stride_(stride),
      breadth_filter_(breadth_filter),
      num_channels_(num_channels){
    breadth_output_ = (breadth_neuron-1)/stride + 1;
    neurons_.resize( breadth_neuron * breadth_neuron * num_channels );
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

  assert( calculated_ );

  for( int i = 0; i < output_neurons.size(); i++ )
    output_neurons[i].u = 0.0;
  
  for( int i = 0; i < breadth_output_; i++ ){
    for( int j = 0; j < breadth_output_; j++ ){
      for( int k = 0; k < num_channels_; k++ ){
	double maxv = -numeric_limits<double>::max();
	for( int p = 0; p < breadth_filter_; p++ ){
	  for( int q = 0; q < breadth_filter_; q++ ){
	    int x = j*stride_+q;
	    int y = i*stride_+p;
	    double z = -numeric_limits<double>::max();
	    if( y < breadth_neuron_ && x < breadth_neuron_ )
	      z = neurons_[k*size2 + y*breadth_neuron_ + x].z;
	    if( maxv < z ){
	      maxv = z;
	      maxid = k*size2 + y*breadth_neuron_ + x;
	    }
	  }
	}
	assert( maxv != -numeric_limits<double>::max() );
	output_neurons[k*size + i*breadth_output_ + j].u = maxv;
      }
    }
  }

  layer->calculated_ = false;
}


/*void PoolLayer::BackPropagate( DoubleVector2d next_deltas ){
  deltas_.resize( next_deltas.size() );

  for( int i = 0; i < e
}*/
