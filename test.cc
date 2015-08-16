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
  int LOOP_N = 100;//25000-2;//1000;
  int namonakiacc = 0;
  vector<double> in, out;
  DoubleVector2d ins, outs;

  unsigned char* pixels;
  int width, height, bpp;
  
  NeuralNet net;
  RectifiedLinear rel;
  LogisticSigmoid sigmoid;
  Softmax softmax;
  
  ConvLayer *conv1 = new ConvLayer(64, 3, 2, 5, 8, &sigmoid, 0.00005);
  PoolLayer *pool1 = new PoolLayer(32, 8, 2, 3);
  ConvLayer *conv2 = new ConvLayer(16, 8, 2, 5, 16, &sigmoid, 0.00005);
  PoolLayer *pool2 = new PoolLayer(8, 16, 4, 3);
  FullyConnectedLayer *full1 = new FullyConnectedLayer(2*2*16, &softmax, 0.00005);
  FullyConnectedLayer *full2 = new FullyConnectedLayer(2, &sigmoid, 0.00005);
  

  srand(time(NULL));
  net.AppendLayer(conv1);
  net.AppendLayer(pool1);
  net.AppendLayer(conv2);
  net.AppendLayer(pool2);
  net.AppendLayer(full1);
  net.AppendLayer(full2);  
  net.ConnectLayers();

  vector<double> testin;
  pixels = stbi_load( "processed/dog.0.jpg" , &width , &height , &bpp , 0 );
  for( int k = 0; k < 3; k++ )
    for( int i = 0; i < height; i++ )
      for( int j = 0; j < width; j++ )
	testin.push_back( (double)pixels[(i*width+j)*3+k] / 256.0 );
  
  for (int o=0; o<1000; o++) {
      for( int loop = 0; loop < LOOP_N; loop++ ){
        in.clear();
        out.clear();
      
        if( /*mt()*/loop % 2 == 0 ){
          out.push_back( 1.0 ); out.push_back( 0.0 );
          sprintf( filename , "processed/cat.%d.jpg" , loop/2/*mt()%MAX_FILE*/ );
        } else {
          out.push_back( 0.0 ); out.push_back( 1.0 );
          sprintf( filename , "processed/dog.%d.jpg" , loop/2/*mt()%MAX_FILE*/ );
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
        /*if (out2[0] > 0.3) {
            conv1->learning_rate_ = 0.000005;
            conv1->learning_rate_ = 0.000005;
            full1->learning_rate_ = 0.000005;
            full2->learning_rate_ = 0.000005;
        }*/
            
        cout << endl;
      }
  }
}



int rev( int x ){
  int res = 0;

  for( int i = 0; i < 4; i++ )
    res += ( (x>>(i*8)) & 255 ) << (24-i*8);

  return res;
}


double img[60000][28*28+1];
int label[60000];

void TestMNIST(){

  int magic_number;
  int N, H, W;
  
  vector<double> in, out;
  DoubleVector2d ins, outs;

  int namonakiacc = 0;  

  NeuralNet net;
  RectifiedLinear rel;
  LogisticSigmoid sigmoid;
  Softmax softmax;

  /*
  ConvLayer *conv1 = new ConvLayer(28, 1, 2, 5, 8, &sigmoid, 0.00005);
  PoolLayer *pool1 = new PoolLayer(14, 8, 2, 3);
  FullyConnectedLayer *full1 = new FullyConnectedLayer(7*7*8, &softmax, 0.00005);
  FullyConnectedLayer *full2 = new FullyConnectedLayer(10, &sigmoid, 0.00005);
  

  srand(time(NULL));
  net.AppendLayer(conv1);
  net.AppendLayer(pool1);
  net.AppendLayer(full1);
  net.AppendLayer(full2);  
  net.ConnectLayers();
  */

  FullyConnectedLayer *full1 = new FullyConnectedLayer(28*28, &sigmoid, 0.1);
  FullyConnectedLayer *full2 = new FullyConnectedLayer(32, &sigmoid, 0.1);  
  FullyConnectedLayer *full3 = new FullyConnectedLayer(10, &sigmoid, 0.1);
  
  srand(time(NULL));
  net.AppendLayer(full1);
  net.AppendLayer(full2);
  net.AppendLayer(full3);    
  net.ConnectLayers();  

  FILE *fp;

  fp = fopen( "train-images-idx3-ubyte" , "rb" );
  
  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);

  fread( &H , sizeof( H ) , 1 , fp );
  H = rev(H);
  
  fread( &W , sizeof( W ) , 1 , fp );
  W = rev(W);


  for( int k = 0; k < N; k++ ){
    if( k % (N/10) == (N/10)-1) cerr << k << " / " << N << endl;
    for( int i = 0; i < H; i++ ){
      for( int j = 0; j < W; j++ ){
	unsigned char tmp;
	fread( &tmp , sizeof( tmp ) , 1 , fp );
	img[k][i*W+j] = (double)tmp / 256.0;
      }
    }
  }

  fclose( fp );
  
  fp = fopen( "train-labels-idx1-ubyte" , "rb" );

  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);

  for( int k = 0; k < N; k++ ){
    unsigned char tmp;
    fread( &tmp , sizeof( tmp ) , 1 , fp );
    label[k] = int(tmp);
  }

  fclose( fp );

  
  for( int loop = 0; loop < N; loop++ ){
    if( loop % 100 == 0 ) cout << loop << " / " << N << endl;
    in.clear();
    out = vector<double>(10,0.0);

    for( int i = 0; i < H; i++ )
      for( int j = 0; j < W; j++ )
	in.push_back( img[loop][i*W+j] );

    out[ label[loop] ] = 1.0;


    ins.clear();
    ins.push_back( in );
    outs.clear();
    outs.push_back( out );
    
    net.TrainNetwork(ins, outs);
  }



  fp = fopen( "t10k-images-idx3-ubyte" , "rb" );
  
  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);

  fread( &H , sizeof( H ) , 1 , fp );
  H = rev(H);
  
  fread( &W , sizeof( W ) , 1 , fp );
  W = rev(W);


  for( int k = 0; k < N; k++ ){
    img[k][0] = 1.0;
    for( int i = 0; i < H; i++ ){
      for( int j = 0; j < W; j++ ){
	unsigned char tmp;
	fread( &tmp , sizeof( tmp ) , 1 , fp );
	img[k][i*W+j] = (double)tmp / 256.0;
      }
    }
  }

  fclose( fp );
  
  fp = fopen( "t10k-labels-idx1-ubyte" , "rb" );

  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);

  for( int k = 0; k < N; k++ ){
    unsigned char tmp;
    fread( &tmp , sizeof( tmp ) , 1 , fp );
    label[k] = int(tmp);
  }

  fclose( fp );


  for( int loop = 0; loop < N; loop++ ){
    if( loop % 100 == 0 ) cout << loop << " / " << N << endl;
    in.clear();
    out = vector<double>(10,0.0);

    for( int i = 0; i < H; i++ )
      for( int j = 0; j < W; j++ )
	in.push_back( img[loop][i*W+j] );

    net.PropagateLayers( in , out );

    int res = 0;
    for( int i = 1; i < 10; i++ )
      if( out[res] < out[i] ) res = i;

    if( res == label[loop] ) namonakiacc++;
  }

  cout << namonakiacc << " / " << N << endl;
}



int main(){
  //TestFullyConnectedLayer();
  //TestPoolLayer();
  //TestDeepLearning();
  TestMNIST();
}
