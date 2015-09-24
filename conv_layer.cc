#include "conv_layer.h"
#include "stb_image_write.h"

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

  layer_type_ = CONV_LAYER;

  num_input_ = breadth_neuron * breadth_neuron;
  assert(num_input_/breadth_neuron == breadth_neuron);
  num_input_ *= num_channels;
  assert(num_input_/num_channels == breadth_neuron * breadth_neuron);

  breadth_output_ = (breadth_neuron-1)/stride + 1;

  num_output_ = breadth_output_ * breadth_output_;
  assert(num_output_/breadth_output_ == breadth_output_);
  num_output_ *= num_filters;
  assert(num_output_/num_filters == breadth_output_ * breadth_output_);

  styleset_ = false;
  contentset_ = false;
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
    w.gsum = 1000.0;
    biases_[i] = w;
  }


  double lim = 0.85 / sqrt( num_channels_*breadth_filter_*breadth_filter_ );
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
	  w.gsum = 1000.0;	  
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

  CalculateStyle( units );
  CalculateContent( units );

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

  
  vector<double> next_delta_with_styleerror = next_delta;
  for( int m = 0; m < num_filters_; m++ ){
    for( int i = 0; i < area_output; i++ ){
      int idx = m * area_output + i;
      next_delta_with_styleerror[idx] += styleerror_[i][m];
    }
  }

  if( contentset_ ){
    assert( content_.size() == next_delta_with_styleerror.size() );
    for( int i = 0; i < content_.size(); i++ )
      next_delta_with_styleerror[i] += content_[i] - tcontent_[i];
    double res = 0.0;
    for( int i = 0; i < content_.size(); i++ )
      res += ( content_[i] - tcontent_[i] ) * ( content_[i] - tcontent_[i] );
    printf( "content : %lf\n" , res );
  }
  double res = 0.0;
  for( int m1 = 0; m1 < num_filters_; m1++ )
    for( int m2 = 0; m2 < num_filters_; m2++ )
      res += ( style_[m1][m2] - tstyle_[m1][m2] ) * ( style_[m1][m2] - tstyle_[m1][m2] );
  printf( "style : %lf\n" , res );
  
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

		/*
		 delta[input_idx] += 
                  next_delta[output_idx] * 
                  weights_[m][k][p][q].val;
		*/
                delta[input_idx] += 
                  next_delta_with_styleerror[output_idx] * 
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
  if( fp == NULL ){
    printf( "%s Not Found\n" , s );
    return;
  }
  printf( "%s Found\n" , s );

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

void ConvLayer::CalculateStyle( vector<struct Neuron> &units ){
  int area_output = breadth_output_ * breadth_output_;
  
  style_.clear();  
  style_.resize( num_filters_ );
  for( int m1 = 0; m1 < num_filters_; m1++ ){
    style_[m1].clear();
    style_[m1].resize( num_filters_ , 0.0 );
    for( int m2 = 0; m2 < num_filters_; m2++ ){
      for( int i = 0; i < area_output; i++ ){
	int idx1 = m1 * area_output + i;
	int idx2 = m2 * area_output + i;
	style_[m1][m2] += units[idx1].z * units[idx2].z;
      }
    }
  }

  if( !styleset_ ) return;

  
  styleerror_.resize( area_output );
  for( int i = 0; i < area_output; i++ ){
    styleerror_[i].resize( num_filters_ );
    for( int m = 0; m < num_filters_; m++ ){
      styleerror_[i][m] = 0;
      for( int k = 0; k < num_filters_; k++ ){
	int idx = k * area_output + i;
	styleerror_[i][m] += units[idx].z * ( style_[k][m] - tstyle_[k][m] );
      }
      styleerror_[i][m] /= 1e2;
    }
  } 

}

void ConvLayer::VisualizeStyle( int filenum ,  int depth ){ 

  int size = num_filters_;
  
  unsigned char pixels[size*size];
  char outputfilename[256];

  double maxv = -1000;
  double minv = 1000;
  
  for( int i = 0; i < size; i++ ){
    for( int j = 0; j < size; j++ ){
      maxv = max( maxv , style_[i][j] );
      minv = min( minv , style_[i][j] );
    }
  }

  
  for( int i = 0; i < size; i++ ){
    for( int j = 0; j < size; j++ ){
      pixels[i*size+j] = (unsigned char)( ( style_[i][j] - minv ) / ( maxv - minv ) * 255 );
      assert( 0 <= pixels[i*size+j] && pixels[i*size+j] < 256 );
    }
  }
  
  sprintf( outputfilename , "output/img_%d_s_%d.png" , filenum , depth );
  stbi_write_png( outputfilename, size, size, 1, pixels, size );
}

void ConvLayer::SetStyle(){
  styleset_ = true;
  tstyle_ = style_;
}


void ConvLayer::CalculateContent( vector<struct Neuron> &units ){
  content_.clear();
  for( int i = 0; i < units.size(); i++ )
    content_.push_back( units[i].z );
}

void ConvLayer::SetContent(){
  contentset_ = true;
  tcontent_ = content_;
}
