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
       f_(f),
       learning_rate_(learning_rate) {
  int num_input;
  int num_output;

  assert(stride >= 1);

  num_input = breadth_neuron * breadth_neuron;
  assert(num_input/breadth_neuron == breadth_neuron);
  num_input *= num_channels;
  assert(num_input/num_channels == breadth_neuron * breadth_neuron);

  num_input_ = num_input_;
  breadth_output_ = (breadth_neuron-1)/stride + 1;

  num_output = breadth_output_ * breadth_output_;
  assert(num_output/breadth_output_ == breadth_output_);
  num_output_ *= num_filters;
  assert(num_output/num_filters == breadth_output_ * breadth_output_);
  num_output_ = num_output;
}

void ConvLayer::CheckInputUnits(vector<struct Neuron> const &units) {
  assert(units.size() == num_input_);
}

void ArrangeOutputUnits(vector<struct Neuron> &units) {
  units.resize(num_output_);
}

void ConvLayer::ConnectNeurons(
    vector<struct Neuron> const &input, 
    vector<struct Neuron> const &output) {
  assert(!neuron_connected_);
  assert(input.size() == num_input_);
  assert(output.size() == num_output_);

  biases_.resize(num_filters_);
  for (int i=0; i<num_filters_; i++) { // Bijm == Bm
    struct Weight w;

    w.val = GenRandom(-0.5, 0.5);
    w.lazy_sub = 0.0;
    w.count = 0;
    biases_[i] = w;
  }

  double lim = 1.0/sqrt(num_input);
  weights_.resize(num_filters_);
  for (int m=0; m<num_filters_; m++) {
    weights_[m].resize(num_channels_);
    for (int k=0; k<num_channels_; k++) {
      weights_[m][k].resize(breadth_filter_);
      for (int i=0; i<breadth_filter_; i++) {
        weights_[m][k][i].resize(breadth_filter_);
        for (int j=0; j<breadth_filter_; j++) {
          struct Weight w;

          w.val = GenRandom(-lim, lim);
          w.lazy_sub = 0.0;
          w.count = 0;
          weights_[m][k][i][j] = w;
        }
      }
    }
  }

  neuron_connected_ = true;
}

void ConvLayer::CalculateOutputUnits(vector<struct Neuron> &units) {
  assert(units.size() == num_output_);

  for (int i=0; i<num_output_; i++) {
    units[i].z = f_(units[i].u);
  }
}


//    3d_neurons[z][y][x] z := m, k, z...
//                        y := i, p, y...
//                        x := j, q, x...
void ConvLayer::Propagate(
    vector<struct Neuron> const &input, 
    vector<struct Neuron> &output) {
  int area_output = breadth_output_ * breadth_output_;
  int area_input = breadth_neuron_ * breadth_neuron_;

  assert(input.size() == num_input_);
  assert(output.size() == num_output_);

  for (int i=0; i<num_output_; i++) { 
    output[i].u = 0.0; 
  }

  assert(biases_.size() == num_filters_);
  assert(weights_.size() == num_filters_);
  for (int m=0; m<num_filters_; m++) { 
    assert(weights[m]_.size() == num_channels_);
    for (int i=0; i<breadth_output_; i++) {
      for (int j=0; j<breadth_output_; j++) {
        int output_idx = m*area_output + i*breadth_output_ + j;
        double sum_conv = 0.0;

        assert(i*stride_ < breadth_neuron_);
        assert(j*stride_ < breadth_neuron_);

        for (int k=0; k<num_channels_; k++) {
          assert(weights[m][k]_.size() == breadth_filter_);
          for (int p=0; p<breadth_filter_; p++) {
            assert(weights[m][k][p]_.size() == breadth_filter_);
            for (int q=0; q<breadth_filter_; q++) {
              int x = j*stride_ + q;
              int y = i*stride_ + p;
              int input_idx = k*area_input + y*breadth_neuron_ + x;
              double z = 0.0;

              // fix the way of padding
              if (x < breadth_neuron_ && y < breadth_neuron_) {
                assert(input_idx < input.size());
                z = input[input_idx].z;
              }
              sum_conv += z * weights_[m][k][p][q].val;
            }
          }
        }

        assert(output_idx < output.size());
        output[output_idx].u = sum_conv + biases_[m];
      }
    }
  }
}
 
void ConvLayer::BackPropagate(
        vector<struct Neuron> const &input, 
        vector<double> const &next_delta, 
        vector<double> &delta) {
    
  int area_output = breadth_output_ * breadth_output_;
  int area_input = breadth_neuron_ * breadth_neuron_;

  assert(input.size() == num_input_);
  assert(delta.size() == num_input_);
  assert(next_delta.size() == num_output_);
  
  for (int i=0; i<num_input_; i++) {
    delta[i] = 0.0;
  }

  assert(weights_.size() == num_filters_);
  for (int m=0; m<num_filters_; m++) { 
    assert(weights_[m].size() == num_channels_);
    for (int i=0; i<breadth_output_; i++) {
      for (int j=0; j<breadth_output_; j++) {
        int output_idx = m*area_output + i*breadth_output_ + j;
        double sum_conv = 0.0;
  
        assert(i*stride_ < breadth_neuron_);
        assert(j*stride_ < breadth_neuron_);
  
    	for (int k=0; k<num_channels_; k++) {
          assert(weights_[m][k].size() == breadth_filter_);
          for (int p=0; p<breadth_filter_; p++) {
            assert(weights[m][k][p]_.size() == breadth_filter_);
            for (int q=0; q<breadth_filter_; q++) {
              int x = j*stride_ + q;
              int y = i*stride_ + p;
              int input_idx = k*area_input + y*breadth_neuron_ + x;
    
              if (x < breadth_neuron_ && y < breadth_neuron_) {
                delta[input_idx] += 
                  next_delta[output_idx] * 
                  weights_[m][k][p][q] * 
                  f_->CalculateDerivative(input[input_idx].u);
    	      }
            }
          }
        }
      }
    }
  }
}

void ConvLayer::UpdateLazySubtrahend(
    vector<struct Neuron> const &input, 
    const vector<double> &next_delta) {
  int area_output = breadth_output_ * breadth_output_;
  int area_input = breadth_neuron_ * breadth_neuron_;
  
  assert(input.size() == num_input_);
  assert(next_delta.size() == num_output_);

  assert(weights_.size() == num_filters_);
  for (int m=0; m<num_filters_; m++) {
    assert(weights_[m].size() == num_channels_);
    for (int i=0; i<breadth_output_; i++) {
  	  for (int j=0; j<breadth_output_; j++) {
        int output_idx = m*area_output + i*breadth_output_ + j;

        assert(i*stride_ < breadth_neuron_);
        assert(j*stride_ < breadth_neuron_);
	    for (int k=0; k<num_channels_; k++) {
          assert(weights_[m][k].size() == breadth_filter_);
	      for (int p=0; p<breadth_filter_; p++) {
            assert(weights_[m][k][p].size() == breadth_filter_);
	        for (int q=0; q<breadth_filter_; q++) {
              int x = j*stride_ + q;
              int y = i*stride_ + p;
		      int input_idx = k*area_input + y*breadth_neuron_ + x;

	          if (x < breadth_neuron_ && y < breadth_neuron_) {
	         	  Weight &w = weights_[m][k][p][q];
              
                assert(output_idx < next_delta.size());
                assert(input_idx < input.size();
                w.lazy_sub += 
                  learning_rate_ * 
                  next_delta[output_idx] * 
                  input[input_idx].z;
                w.count++; // think later
              }
            }
          }
        }
      }
    }
  }

  assert(biases_.size() == num_filter_);
  for (int m=0; m<num_filters_; m++) {
    for (int i=0; i<breadth_output_; i++) {
  	  for (int j=0; j<breadth_output_; j++) {
        int output_idx = m*area_output + i*breadth_output_ + j;
        struct Weight &w = biases_[m];
          
        assert(output_idx < next_delta.size());
        w.lazy_sub += next_delta[output_idx];
        w.count++;
      }
    }
  }
}
                
void ConvLayer::ApplyLazySubtrahend() {
  assert(weights_.size() == num_filters_);
  for (int m=0; m<num_filters_; m++) {
    assert(weights_[m].size() == num_channels_);
	for (int k=0; k<num_channels_; k++) {
      assert(weights_[m][k].size() == breadth_filter_);
	  for (int p=0; p<breadth_filter_; p++) {
        assert(weights_[m][k][p].size() == breadth_filter_);
	    for (int q=0; q<breadth_filter_; q++) {
          struct Weight &w = weights_[m][k][p][q];

          assert(w.count > 0);
          w.val -= w.lazy_sub / w.count;
          w.lazy_sub = 0.0;
          w.count = 0;
        }
      }
    }
  }

  assert(biases_.size() == num_filter_);
  for (int m=0; m<num_filters_; m++) {
      struct Weight &w = biases_[m];

      assert(w.count > 0);
      w.val -= w.lazy_sub / w.count;
      w.lazy_sub = 0.0;
      w.count = 0;
  }
}
