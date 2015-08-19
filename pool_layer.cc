#include "pool_layer.h"
#include "iostream"

PoolLayer::PoolLayer(
  int breadth_neuron, 
  int num_channels, 
  int stride, 
  int breadth_filter,
  ActivationFunction *f )
    : neuron_connected_(false),
      propagated_(false),
      breadth_neuron_(breadth_neuron), 
      stride_(stride),
      breadth_filter_(breadth_filter),
      num_channels_(num_channels),
      Layer(f){
  int num_input;
  int num_output;

  breadth_output_ = (breadth_neuron-1)/stride + 1;

  num_input = breadth_neuron * breadth_neuron;
  assert(num_input / breadth_neuron == breadth_neuron);
  num_input *= num_channels;
  assert(num_input / num_channels == breadth_neuron * breadth_neuron);
  num_input_ = num_input;

  num_output =breadth_output_ * breadth_output_;
  assert(num_output / breadth_output_ == breadth_output_);
  num_output *= num_channels;
  assert(num_output / num_channels == breadth_output_ * breadth_output_);
  num_output_ = num_output;
}

void PoolLayer::CheckInputUnits(vector<struct Neuron> const &units) {
  assert(units.size() == num_input_);
}

void PoolLayer::ArrangeOutputUnits(vector<struct Neuron> &units) {
  units.resize(num_output_);
}

void PoolLayer::ConnectNeurons(
    vector<struct Neuron> const &input,
    vector<struct Neuron> const &output) {
  assert(!neuron_connected_);
  assert(input.size() == num_input_);
  assert(output.size() == num_output_);

  maxid.resize(num_output_);

  neuron_connected_ = true;
}

void PoolLayer::CalculateOutputUnits(vector<struct Neuron> &units) {
  assert(units.size() == num_output_);

  double outputmax = -1000;
  double outputmin = 1000;
  
  for (int i=0; i<num_output_; i++) {
    units[i].z = f_->Calculate(units[i].u, units);

    outputmax = max( outputmax , units[i].z );
    outputmin = min( outputmin , units[i].z );    
  }

  //printf( "poolsig : %lf %lf\n" , outputmax , outputmin );
  
  // do nothing
}

void PoolLayer::Propagate(
			  vector<struct Neuron> const &input,
			  vector<struct Neuron> &output) {
  int area_output = breadth_output_ * breadth_output_;
  int area_input = breadth_neuron_ * breadth_neuron_;

  assert(input.size() == num_input_);
  assert(output.size() == num_output_);

  double outputmax = -1000;
  double outputmin = 1000;
  
  for (int i=0; i<num_output_; i++) {
    output[i].u = 0.0;
  }

  assert(maxid.size() == num_output_);
  for (int i=0; i<breadth_output_; i++) {
    for (int j=0; j<breadth_output_; j++) {
      for (int k=0; k<num_channels_; k++) {
        int output_idx = k*area_output + i*breadth_output_ + j;
	double maxv = -numeric_limits<double>::max();

        assert(j*stride_ < breadth_neuron_);
        assert(i*stride_ < breadth_neuron_);

    	for (int p=0; p<breadth_filter_; p++) {
	  for (int q=0; q<breadth_filter_; q++) {
    	    int x = j*stride_+q;
	    int y = i*stride_+p;
	    double z = -numeric_limits<double>::max();
            int input_idx = k*area_input + y*breadth_neuron_ + x;

	    if (x < breadth_neuron_ && y < breadth_neuron_) {
	      z = input[input_idx].z;
            }

    	    if (maxv < z) {
	      maxv = z;
	      maxid[output_idx] = input_idx;
	    }
          }
	}

	assert(maxv != -numeric_limits<double>::max());
	output[output_idx].u = maxv;

	outputmax = max( outputmax , maxv );
	outputmin = min( outputmin , maxv );
      }
    }
  }

  //printf( "pool : %lf %lf\n" , outputmax , outputmin );

  for( int i = 0; i < input.size(); i++ )
    assert( input[i].z == output[i].u );

  
  propagated_ = true;
}

void PoolLayer::BackPropagate(
    vector<struct Neuron> const &input,
    vector<double> const &next_delta,
    ActivationFunction *f,
    vector<double> &delta) {
  assert(propagated_);
  assert(input.size() == num_input_);
  assert(next_delta.size() == num_output_);
  assert(maxid.size() == num_output_);

  delta.resize(num_input_);
  for (int i=0; i<num_input_; i++) {
    delta[i] = 0.0;
  }

  double deltamax = -1000;
  double deltamin = 1000;
  
  for (int i=0; i<num_output_; i++) {
    delta[maxid[i]] += next_delta[i] * f->CalculateDerivative(input[maxid[i]].u);

    deltamax = max( deltamax , delta[maxid[i]] );
    deltamin = min( deltamin , delta[maxid[i]] );
  }

  //printf( "pooldelta : %lf %lf\n" , deltamax , deltamin );
  
  propagated_ = false;
}

void PoolLayer::UpdateLazySubtrahend(
    vector<struct Neuron> const &input,
    const vector<double> &next_delta) {
  assert(input.size() == num_input_);
  assert(next_delta.size() == num_output_);

  // do nothing
}

void PoolLayer::ApplyLazySubtrahend() {
  // do nothing
}
