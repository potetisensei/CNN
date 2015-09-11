#include "conv_layer.h"

ConvLayer::ConvLayer(
  int breadth_neuron, 
  int num_channels, 
  int stride,
  int padding,
  int breadth_filter, 
  int num_filters, 
  ActivationFunction *f, 
  double learning_rate,
  double momentum,
  double dropout_rate)
    :  neuron_connected_(false),
       breadth_neuron_(breadth_neuron), 
       num_channels_(num_channels), 
       stride_(stride),
       padding_(padding),
       breadth_filter_(breadth_filter),
       num_filters_(num_filters),
       learning_rate_(learning_rate),
       momentum_(momentum),
       Layer(f, dropout_rate) {
  assert(stride >= 1);

  num_input_ = breadth_neuron * breadth_neuron;
  assert(num_input_/breadth_neuron == breadth_neuron);
  num_input_ *= num_channels;
  assert(num_input_/num_channels == breadth_neuron * breadth_neuron);

  breadth_output_ = (breadth_neuron-1)/stride + 1;

  num_output_ = breadth_output_ * breadth_output_;
  assert(num_output_/breadth_output_ == breadth_output_);
  num_output_ *= num_filters;
  assert(num_output_/num_filters == breadth_output_ * breadth_output_);
}

void ConvLayer::CheckInputUnits(vector<struct Neuron> const &units) {
  assert(units.size() == num_input_);
}

void ConvLayer::ArrangeOutputUnits(vector<struct Neuron> &units) {
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

    w.val = GenRandom(0, 0.1);
    w.lazy_sub = 0.0;
    w.count = 0;
    w.gsum = 0.0;
    biases_[i] = w;
  }


  double lim = 1.0 / ( num_channels_*breadth_filter_*breadth_filter_ ); //1.0 / sqrt( num_channels_*breadth_filter_*breadth_filter_ );
  weights_.resize(num_filters_);
  for (int m=0; m<num_filters_; m++) {
    weights_[m].resize(num_channels_);
    for (int k=0; k<num_channels_; k++) {
      weights_[m][k].resize(breadth_filter_);
      for (int i=0; i<breadth_filter_; i++) {
        weights_[m][k][i].resize(breadth_filter_);
        for (int j=0; j<breadth_filter_; j++) {
          struct Weight w;

          w.val = GenRandom(0, lim);
          w.lazy_sub = 0.0;
          w.count = 0;
	  w.gsum = 0.0;	  
          weights_[m][k][i][j] = w;
        }
      }
    }
  }

  neuron_connected_ = true;
}

void ConvLayer::CalculateOutputUnits(vector<struct Neuron> &units) {
  assert(units.size() == num_output_);

  double outputmax = -1000;
  double outputmin = 1000;
  
  for (int i=0; i<num_output_; i++) {
    units[i].z = f_->Calculate(units[i].u, units);

    outputmax = max( outputmax , units[i].z );
    outputmin = min( outputmin , units[i].z );    
  }

#if DEBUG
  printf( "convsig : %lf %lf\n" , outputmax , outputmin );
#endif
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

  double outputmax = -1000;
  double outputmin = 1000;

  assert(biases_.size() == num_filters_);
  assert(weights_.size() == num_filters_);
  for (int m=0; m<num_filters_; m++) { 
    assert(weights_[m].size() == num_channels_);
    for (int i=0; i<breadth_output_; i++) {
      for (int j=0; j<breadth_output_; j++) {
        int output_idx = m*area_output + i*breadth_output_ + j;
        double sum_conv = 0.0;

        assert(i*stride_ - padding_ < breadth_neuron_);
        assert(j*stride_ - padding_ < breadth_neuron_);

        for (int k=0; k<num_channels_; k++) {
          assert(weights_[m][k].size() == breadth_filter_);
          for (int p=0; p<breadth_filter_; p++) {
            assert(weights_[m][k][p].size() == breadth_filter_);
            for (int q=0; q<breadth_filter_; q++) {
              int x = j*stride_ - padding_ + q;
              int y = i*stride_ - padding_ + p;
              int input_idx = k*area_input + y*breadth_neuron_ + x;

              // Fix the way of padding
              if ( 0 <= x && x < breadth_neuron_ && 0 <= y && y < breadth_neuron_ ) {
                assert(input_idx < input.size());
		sum_conv += input[input_idx].z * weights_[m][k][p][q].val;
              }

            }
          }
        }

        assert(output_idx < output.size());
        output[output_idx].u = sum_conv + biases_[m].val;
	
	outputmax = max( outputmax, output[output_idx].u );
	outputmin = min( outputmin, output[output_idx].u );		
      }
    }
  }
  
#if DEBUG
  printf( "conv : %lf %lf\n" , outputmax , outputmin );
#endif
}
 
void ConvLayer::BackPropagate(
    vector<struct Neuron> const &input, 
    vector<double> const &next_delta, 
    ActivationFunction *f,
    vector<double> &delta) {

  int area_output = breadth_output_ * breadth_output_;
  int area_input = breadth_neuron_ * breadth_neuron_;

  assert(input.size() == num_input_);
  assert(next_delta.size() == num_output_);
  
  delta.resize(num_input_);
  for (int i=0; i<num_input_; i++) {
    delta[i] = 0.0;
  }

  double deltamax = -1000;
  double deltamin = 1000;
  
  assert(weights_.size() == num_filters_);
  for (int m=0; m<num_filters_; m++) { 
    assert(weights_[m].size() == num_channels_);
    for (int i=0; i<breadth_output_; i++) {
      for (int j=0; j<breadth_output_; j++) {
        int output_idx = m*area_output + i*breadth_output_ + j;
        double sum_conv = 0.0;
  
        assert(i*stride_ - padding_ < breadth_neuron_);
        assert(j*stride_ - padding_ < breadth_neuron_);
  
    	for (int k=0; k<num_channels_; k++) {
          assert(weights_[m][k].size() == breadth_filter_);
          for (int p=0; p<breadth_filter_; p++) {
            assert(weights_[m][k][p].size() == breadth_filter_);
            for (int q=0; q<breadth_filter_; q++) {
              int x = j*stride_ - padding_ + q;
              int y = i*stride_ - padding_ + p;
              int input_idx = k*area_input + y*breadth_neuron_ + x;
    
              if ( 0 <= x && x < breadth_neuron_ && 0 <= y && y < breadth_neuron_ && input[input_idx].u > 0) {
		assert( 0 <= input_idx && input_idx < delta.size() );

                delta[input_idx] += 
                  next_delta[output_idx] * 
                  weights_[m][k][p][q].val;

    	      }
            }
          }
        }
      }
    }
  }


#if DEBUG
  for( int i = 0; i < delta.size(); i++ ){
    deltamax = max( deltamax , delta[i] );
    deltamin = min( deltamin , delta[i] );
  }
  printf( "convdelta : %lf %lf\n" , deltamax , deltamin );
#endif
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
    for (int k=0; k<num_channels_; k++) {
      assert(weights_[m][k].size() == breadth_filter_);
      for (int p=0; p<breadth_filter_; p++) {
	assert(weights_[m][k][p].size() == breadth_filter_);
	for (int q=0; q<breadth_filter_; q++) {
	  Weight &w = weights_[m][k][p][q];

	  w.count++;

	  for (int i=0; i<breadth_output_; i++) {
	    assert(i*stride_ - padding_ < breadth_neuron_);
	    
	    for (int j=0; j<breadth_output_; j++) {
	      assert(j*stride_ - padding_ < breadth_neuron_);	      

              int x = j*stride_ - padding_ + q;
              int y = i*stride_ - padding_ + p;
	      int output_idx = m*area_output + i*breadth_output_ + j;
	      int input_idx = k*area_input + y*breadth_neuron_ + x;

	      if (0 <= x && x < breadth_neuron_ && 0 <= y && y < breadth_neuron_) {

#if DEBUG
                assert(output_idx < next_delta.size());
                assert(input_idx < input.size());
#endif
                w.lazy_sub += 
                  next_delta[output_idx] * 
                  input[input_idx].z;
              }
            }
	  }
        }
      }
    }
  }


  assert(biases_.size() == num_filters_);
  for (int m=0; m<num_filters_; m++) {
    struct Weight &w = biases_[m];

    w.count++;

    for (int i=0; i<breadth_output_; i++) {
      for (int j=0; j<breadth_output_; j++) {
        int output_idx = m*area_output + i*breadth_output_ + j;

        assert(output_idx < next_delta.size());
        w.lazy_sub += next_delta[output_idx];
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

	  if( MOMENTUM ){
	     double prevdelta = - w.lazy_sub * learning_rate_ / w.count + momentum_ * w.gsum;
	     w.val += prevdelta;
	     w.gsum = prevdelta;
	  } else if( ADAGRAD ){
	    w.gsum += (w.lazy_sub / w.count) * (w.lazy_sub / w.count);
	    w.val -= learning_rate_ / ( sqrt( w.gsum ) + 1.0 ) * w.lazy_sub / w.count;
	  }
	  
          w.lazy_sub = 0.0;
          w.count = 0;
        }
      }
    }
  }

  assert(biases_.size() == num_filters_);

  for (int m=0; m<num_filters_; m++) {
    struct Weight &w = biases_[m];

    assert(w.count > 0);

    if( MOMENTUM ){
       double prevdelta = - w.lazy_sub * learning_rate_ / w.count + momentum_ * w.gsum;
       w.val += prevdelta;
       w.gsum = prevdelta;
    } else if( ADAGRAD ){
      w.gsum += (w.lazy_sub / w.count) * (w.lazy_sub / w.count);
      w.val -= learning_rate_ / ( sqrt( w.gsum ) + 1.0 ) * w.lazy_sub / w.count;
    }
    
    w.lazy_sub = 0.0;
    w.count = 0;
  }

}


void ConvLayer::Save( char *s ){
  FILE *fp = fopen( s , "w" );
  assert( fp != NULL );  

  assert( num_filters_ == biases_.size() );
  for( int i=0; i<num_filters_; i++ )
    fprintf( fp , "%lf %lf " , biases_[i].val , biases_[i].gsum );
  fprintf( fp , "\n" );

  assert( weights_.size() == num_filters_ );
  for (int m=0; m<num_filters_; m++) {
    assert( weights_[m].size() == num_channels_ );
    for (int k=0; k<num_channels_; k++) {
      assert( weights_[m][k].size() == breadth_filter_ );
      for (int i=0; i<breadth_filter_; i++) {
	assert( weights_[m][k][i].size() == breadth_filter_ );
        for (int j=0; j<breadth_filter_; j++) {
	  fprintf( fp , "%lf %lf " , weights_[m][k][i][j].val , weights_[m][k][i][j].gsum );
        }
      }
      fprintf( fp , "\n" );
    }
  }
  
  fclose( fp );
}

void ConvLayer::Load( char *s ){
  FILE *fp = fopen( s , "r" );
  assert( fp != NULL );  

  assert( num_filters_ == biases_.size() );
  for( int i=0; i<num_filters_; i++ )
    fscanf( fp , "%lf %lf" , &biases_[i].val , &biases_[i].gsum );


  assert( weights_.size() == num_filters_ );
  for (int m=0; m<num_filters_; m++) {
    assert( weights_[m].size() == num_channels_ );
    for (int k=0; k<num_channels_; k++) {
      assert( weights_[m][k].size() == breadth_filter_ );
      for (int i=0; i<breadth_filter_; i++) {
	assert( weights_[m][k][i].size() == breadth_filter_ );
        for (int j=0; j<breadth_filter_; j++) {
	  fscanf( fp , "%lf %lf" , &weights_[m][k][i][j].val , &weights_[m][k][i][j].gsum );
        }
      }
    }
  }
  
  fclose( fp );

}
