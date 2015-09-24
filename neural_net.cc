#include "neural_net.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


NeuralNet::NeuralNet() : layer_connected_(false)  {
    all_neurons_.resize(1);
}

void NeuralNet::SetInputSize(int size) {
    assert(!layer_connected_);
    assert(all_neurons_.size() > 0);
    all_neurons_[0].resize(size);
}

void NeuralNet::AppendLayer(Layer *layer) {
    assert(!layer_connected_);
    layers_.push_back(layer);
    learning_f_.push_back( true );
}

void NeuralNet::ConnectLayers() {
    assert(!layer_connected_);
    assert(layers_.size() > 0);

    all_neurons_.resize(layers_.size()+1);
    for (int i=0; i<layers_.size(); i++) {
        layers_[i]->CheckInputUnits(all_neurons_[i]);
        layers_[i]->ArrangeOutputUnits(all_neurons_[i+1]);
        layers_[i]->ConnectNeurons(all_neurons_[i], all_neurons_[i+1]);
    }
    layer_connected_ = true;
}

void NeuralNet::PropagateLayers(
        vector<double> &input, 
        vector<double> &output,
	bool is_learning ) {
    assert(layer_connected_);
    assert(all_neurons_.size() == layers_.size()+1);

    vector<struct Neuron> &first_neurons = all_neurons_[0];
    vector<struct Neuron> &last_neurons = all_neurons_[layers_.size()];

    assert(first_neurons.size() == input.size());
    assert(last_neurons.size() == output.size());

    for (int i=0; i<input.size(); i++) {
        first_neurons[i].u = input[i];      
        first_neurons[i].z = input[i];
    }

    for (int i=0; i<layers_.size(); i++) {
      if( is_learning ){
	for( int j = 0; j < all_neurons_[i].size(); j++ )
	  all_neurons_[i][j].z *= all_neurons_[i][j].is_valid;
      } else {
	for( int j = 0; j < all_neurons_[i].size(); j++ )
	  all_neurons_[i][j].z *= layers_[i]->dropout_rate_;
      }
      layers_[i]->Propagate(all_neurons_[i], all_neurons_[i+1]);
      layers_[i]->CalculateOutputUnits(all_neurons_[i+1]);
    }

    for (int i=0; i<output.size(); i++) {
        output[i] = last_neurons[i].z;
    }
}

void NeuralNet::BackPropagateLayers(vector<double> &expected) {
    assert(layer_connected_);

    vector<double> delta;
    vector<double> prev_delta;
    int last_idx = layers_.size();
    assert(last_idx+1 == all_neurons_.size());
    vector<struct Neuron> &last_neurons = all_neurons_[last_idx];

    assert(last_neurons.size() == expected.size());
    
    delta.resize(last_neurons.size());
    for (int i=0; i<last_neurons.size(); i++) {
      delta[i] = 0;//last_neurons[i].z - expected[i];
    }

    for (int i=last_idx-1; i>=1; i--) {
         layers_[i]->UpdateLazySubtrahend(all_neurons_[i], delta);
         layers_[i]->BackPropagate(all_neurons_[i], delta, layers_[i-1]->f_, prev_delta);
         delta = prev_delta;
	 
	 assert( delta.size() == all_neurons_[i].size() );
	 for( int j = 0; j < delta.size(); j++ )
	   delta[j] *= all_neurons_[i][j].is_valid;
    }
    layers_[0]->UpdateLazySubtrahend(all_neurons_[0], delta);

    Identity id;
    layers_[0]->BackPropagate(all_neurons_[0], delta, &id, prev_delta);
    delta = prev_delta;
	 
    assert( delta.size() == all_neurons_[0].size() );
    for( int j = 0; j < delta.size(); j++ )
      delta[j] *= all_neurons_[0][j].is_valid;    
    delta_ = delta;
}

void NeuralNet::TrainNetwork(DoubleVector2d &inputs, DoubleVector2d &expected_outputs) {
    assert(layer_connected_);

    vector<double> tmp;
    int last_idx = layers_.size();
    assert(last_idx+1 == all_neurons_.size());
    vector<struct Neuron> &last_neurons = all_neurons_[last_idx];

    tmp.resize(last_neurons.size());
    assert(inputs.size() == expected_outputs.size());
    
    for (int i=0; i<last_idx; i++) {
      layers_[i]->ChooseDropoutUnits(all_neurons_[i]);
    }

    for (int i=0; i<inputs.size(); i++) {
      PropagateLayers(inputs[i], tmp, true);
      BackPropagateLayers(expected_outputs[i]);
    }

    for (int i=last_idx-1; i>=0; i--) {
      if( learning_f_[i] )
	layers_[i]->ApplyLazySubtrahend();
    }
}

void NeuralNet::Save( string s ){
  char filename[256];
  for( int i = 0; i < layers_.size(); i++ ){
    sprintf( filename , "%sdat/dat_%d" , s.c_str() , i );
    layers_[i]->Save( filename );
  }
}

void NeuralNet::Load( string s ){
  char filename[256];
  for( int i = 0; i < layers_.size(); i++ ){
    sprintf( filename , "%sdat/dat_%d" , s.c_str() , i );
    layers_[i]->Load( filename );
  }
}


void NeuralNet::Visualize( int filenum , int depth , int size , int channel_n ){
  
  unsigned char pixels[size*size];
  char outputfilename[256];

  double maxv = -1000;
  double minv = 1000;
  
  for( int i = 0; i < size; i++ ){
    for( int j = 0; j < size; j++ ){
      int id = size*size*channel_n + i*size + j;
      maxv = max( maxv , all_neurons_[depth][id].z );
      minv = min( minv , all_neurons_[depth][id].z );
    }
  }

  
  for( int i = 0; i < size; i++ ){
    for( int j = 0; j < size; j++ ){
      int id = size*size*channel_n + i*size + j;
      pixels[i*size+j] = (unsigned char)( ( all_neurons_[depth][id].z - minv ) / ( maxv - minv ) * 255 );
      assert( 0 <= pixels[i*size+j] && pixels[i*size+j] < 256 );
    }
  }
  
  sprintf( outputfilename , "output/img_%d_%d_%d.png" , filenum , depth , channel_n );
  stbi_write_png( outputfilename, size, size, 1, pixels, size );
}

void NeuralNet::VisualizeStyle( int filenum ){
  for( int i = 0; i < layers_.size(); i++ )
    if( layers_[i]->layer_type_ == CONV_LAYER ) layers_[i]->VisualizeStyle( filenum , i );
}

void NeuralNet::SetLearningFlag( int layer_n , bool f ){
  learning_f_[layer_n] = f;
}

void NeuralNet::SetStyle(){
  for (int i=0; i<layers_.size(); i++)
    if( layers_[i]->layer_type_ == CONV_LAYER )
      layers_[i]->SetStyle();
}

void NeuralNet::SetMiddle( int depth ){
  assert( layers_[depth]->layer_type_ == CONV_LAYER );
  layers_[depth]->SetContent();
}
