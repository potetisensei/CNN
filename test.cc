#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <ctime>
#include "util.h"
#include "neural_net.h"
#include "fully_connected_layer.h"
#include "conv_layer.h"
#include "activation_function.h"
#include "pool_layer.h"
#include "bmp.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

void TestFullyConnectedLayer() {
    NeuralNet net;
    TangentSigmoid tanh;
    Softmax softmax;
    vector<double> input;
    vector<double> output;

    srand(time(NULL));
    net.AppendLayer(new FullyConnectedLayer(3, &tanh, 0.0005));
    net.AppendLayer(new FullyConnectedLayer(2, &tanh, 0.0005));
    net.AppendLayer(new FullyConnectedLayer(3, &softmax, 0.0005));
    net.ConnectLayers();

    for (int j=0; j<10000; j++) {
        DoubleVector2d inputs;
        DoubleVector2d outputs;

        inputs.resize(1);
        outputs.resize(1);
        input.resize(3, 0);
        output.resize(3);
        for (int i=0; i<3; i++) {
            fill(input.begin(), input.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);
            input[i] = 1.0;
            output[i] = 1.0;
            inputs[0] = input;
            outputs[0] = output;
            net.TrainNetwork(inputs, outputs);
            printf("%d: \n", j);
            printf("input: ");
            for (int k=0; k<3; k++) {
                printf("%f ", input[k]);
            }puts("");
            printf("expect: ");
            for (int k=0; k<3; k++) {
                printf("%f ", output[k]);
            }
            puts("");
            net.PropagateLayers(input, output);
            printf("output: ");
            for (int k=0; k<3; k++) {
                printf("%f ", output[k]);
            }
            puts("");
        }
    }
}

void TestConvLayer() {
    BitMapProcessor bmp;
    NeuralNet net;
    RectifiedLinear rel;
    LogisticSigmoid sigmoid;
    Softmax softmax;
    vector<double> input;
    vector<double> output;
    ConvLayer *cl = new ConvLayer(128, 3, 1, 9, 1, &sigmoid, 0.0005);

    srand(time(NULL));
    net.AppendLayer(cl);
    net.AppendLayer(new FullyConnectedLayer(128*128, &sigmoid, 0.0005));
    net.ConnectLayers();
        
    bmp.loadData("lena.bmp");
    assert(bmp.height() == 128);
    assert(bmp.width() == 128);
    input.resize(128*128*3);
    for (int i=0; i<128; i++) {
        for (int j=0; j<128; j++) {
            input[i*128 + j] = bmp.getColor(j, i).r/256.0; 
            input[128*128 + i*128 + j] = bmp.getColor(j, i).g/256.0; 
            input[2*128*128 + i*128 + j] = bmp.getColor(j, i).b/256.0; 
        }
    }
    output.resize(128*128);
    net.PropagateLayers(input, output);
    double maxval = -1.0;
    for (int i=0; i<128; i++) {
        for (int j=0; j<128; j++) {
            maxval = max(maxval, output[i*128+j]);
        }
    }

    for (int i=0; i<128; i++) {
        for (int j=0; j<128; j++) {
            printf("%f ", output[i*128+j]);
            int val = (int)(output[i*128 + j]*256/maxval);
            bmp.setColor(j, i, val, val, val);
        }puts("");
    }
    bmp.writeData("output.bmp");
}

void TestPoolLayer() {
    BitMapProcessor bmp;
    NeuralNet net;
    RectifiedLinear rel;
    LogisticSigmoid sigmoid;
    Softmax softmax;
    vector<double> input;
    vector<double> output;
    ConvLayer *cl = new ConvLayer(128, 3, 1, 9, 1, &sigmoid, 0.0005);
    PoolLayer *pl = new PoolLayer(128, 1, 1, 3);
    
    srand(time(NULL));
    net.AppendLayer(cl);
    net.AppendLayer(pl);
    net.AppendLayer(new FullyConnectedLayer(128*128, &sigmoid, 0.0005));
    net.ConnectLayers();
        
    bmp.loadData("lena.bmp");
    assert(bmp.height() == 128);
    assert(bmp.width() == 128);
    input.resize(128*128*3);
    for (int i=0; i<128; i++) {
        for (int j=0; j<128; j++) {
            input[i*128 + j] = bmp.getColor(j, i).r/256.0; 
            input[128*128 + i*128 + j] = bmp.getColor(j, i).g/256.0; 
            input[2*128*128 + i*128 + j] = bmp.getColor(j, i).b/256.0; 
        }
    }
    output.resize(128*128);
    net.PropagateLayers(input, output);
    double maxval = -1.0;
    for (int i=0; i<128; i++) {
        for (int j=0; j<128; j++) {
            maxval = max(maxval, output[i*128+j]);
        }
    }

    for (int i=0; i<128; i++) {
        for (int j=0; j<128; j++) {
            printf("%f ", output[i*128+j]);
            int val = (int)(output[i*128 + j]*256/maxval);
            bmp.setColor(j, i, val, val, val);
        }puts("");
    }
    bmp.writeData("output.bmp");
}


void TestDeepLearning(){

  mt19937 mt( time( NULL ) );
  int MAX_FILE = 12499;
  char filename[256];
  int LOOP_N = 1000;
  int namonakiacc = 0;
  vector<double> in, out;
  DoubleVector2d ins, outs;

  unsigned char* pixels;
  int width, height, bpp;
  
  NeuralNet net;
  RectifiedLinear rel;
  LogisticSigmoid sigmoid;
  Softmax softmax;

  srand(time(NULL));
  net.AppendLayer(new ConvLayer(128, 3, 1, 5, 8, &sigmoid, 0.00005));
  net.AppendLayer(new PoolLayer(128, 8, 2, 3));
  net.AppendLayer(new ConvLayer(64, 8, 1, 5, 16, &sigmoid, 0.00005));
  net.AppendLayer(new PoolLayer(64, 16, 2, 3));
  net.AppendLayer(new FullyConnectedLayer(32*32*16, &softmax, 0.00005));
  net.AppendLayer(new FullyConnectedLayer(2, &sigmoid, 0.00005));  
  net.ConnectLayers();

  vector<double> testin;
  pixels = stbi_load( "processed/dog.12499.jpg" , &width , &height , &bpp , 0 );
  for( int k = 0; k < 3; k++ )
    for( int i = 0; i < height; i++ )
      for( int j = 0; j < width; j++ )
	testin.push_back( (double)pixels[(i*width+j)*3+k] / 256.0 );
  
  
  for( int loop = 0; loop < LOOP_N; loop++ ){
    in.clear();
    out.clear();
  
    if( mt() % 2 == 0 ){
      out.push_back( 1.0 ); out.push_back( 0.0 );
      sprintf( filename , "processed/cat.%d.jpg" , mt()%MAX_FILE );
    } else {
      out.push_back( 0.0 ); out.push_back( 1.0 );
      sprintf( filename , "processed/dog.%d.jpg" , mt()%MAX_FILE );
    }
  
    pixels = stbi_load( filename , &width , &height , &bpp , 0 );

    for( int k = 0; k < 3; k++ )
      for( int i = 0; i < height; i++ )
	for( int j = 0; j < width; j++ )
	  in.push_back( (double)pixels[(i*width+j)*3+k] / 256.0 );

    ins.clear();
    ins.push_back( in );
    outs.clear();
    outs.push_back( out );

    net.TrainNetwork(ins, outs);
    vector<double> out2(2);

    net.PropagateLayers( testin , out2 );

    cout << filename << endl;
    cout << "o: " << out2[0] << " " << out2[1] << endl;
    cout << endl;
  }

}

int main() {
  //TestFullyConnectedLayer();
  //TestPoolLayer();
  TestDeepLearning();
}
