#include "pool_layer.h"
#include "iostream"

PoolLayer::PoolLayer(
  int breadth_neuron, 
  int num_channels, 
  int stride, 
  int breadth_filter)
    : neuron_connected_(false),
      propagated_(false),
      breadth_neuron_(breadth_neuron), 
      stride_(stride),
      breadth_filter_(breadth_filter),
      num_channels_(num_channels) {
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

void CheckInputUnits(vector<struct Neuron> const &units) {
  assert(units.size() == num_input_);
}

void ArrangeOutputUnits(vector<struct Neuron> &units) {
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

void FullyConnectedLayer::CalculateOutputUnits(vector<struct Neuron> &units) {
    assert(units.size() == num_output_);

    // do nothing
}

void FullyConnectedLayer::Propagate(
    vector<struct Neuron> const &input,
    vector<struct Neuron> &output) {
  int area_output = breadth_output_ * breadth_output_;
  int area_input = breadth_neuron_ * breadth_neuron_;

  assert(input.size() == num_input_);
  assert(output.size() == num_output_);

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
	  }
    }
  }

  propagated_ = true;
}

void ConvLayer::BackPropagate(
    vector<struct Neuron> const &input,
    vector<double> const &next_delta,
    vector<double> &delta) {
  assert(propagated_);
  assert(input.size() == num_input_);
  assert(delta.size() == num_input_);
  assert(next_delta.size() == num_output_);
  assert(maxid.size() == num_output_);

  for (int i=0; i<num_input_; i++) {
    delta[i] = 0.0;
  }

  for (int i=0; i<num_output_; i++) {
    delta[maxid[i]] += next_deltas[i];
  }

  propagated_ = false;
}

void ConvLayer::UpdateLazySubtrahend(
    vector<struct Neuron> const &input,
    const vector<double> &next_delta) {
  assert(input.size() == num_input_);
  assert(next_delta.size() == num_output_);

  // do nothing
}

void ConvLayer::ApplyLazySubtrahend() {
  // do nothing
}
