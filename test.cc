#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <ctime>
#include <random>
#include <time.h>
#include "util.h"
#include "neural_net.h"
#include "fully_connected_layer.h"
#include "conv_layer.h"
#include "activation_function.h"
#include "pool_layer.h"
#include "bmp.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "omp.h"

using namespace std;

void TestFullyConnectedLayer() {
    NeuralNet net;
    RectifiedLinear rel;    
    TangentSigmoid tanh;
    LogisticSigmoid sigmoid;    
    Softmax softmax;
    Identity id;    
    vector<double> input;
    vector<double> output;

    srand(time(NULL));
    net.SetInputSize(4);
    net.AppendLayer(new ConvLayer(2, 1, 1, 0, 2, 4, &rel, 0.1, 0.9, 0.9));
    net.AppendLayer(new PoolLayer(2, 4, 1, 1, &id, 1.0));
    net.AppendLayer(new ConvLayer(2, 4, 1, 0, 2, 16, &rel, 0.1, 0.9, 0.5));
    net.AppendLayer(new PoolLayer(2, 16, 2, 2, &id, 1.0));    
    net.AppendLayer(new FullyConnectedLayer(16, 4, &softmax, 0.1, 0.9, 0.5));
    net.ConnectLayers();

    
    for (int j=0; j<10000; j++) {
        DoubleVector2d inputs;
        DoubleVector2d outputs;

        inputs.resize(2);
        outputs.resize(2);
        input.resize(4, 0);
        output.resize(4);
        for (int i=0; i<4; i++) {
            fill(input.begin(), input.end(), 0.0);
            fill(output.begin(), output.end(), 0.0);
            input[i] = 1.0;
            output[i] = 1.0;
            inputs[0] = input;
            inputs[1] = input;
            outputs[0] = output;
            outputs[1] = output;
            net.TrainNetwork(inputs, outputs);
            printf("%d: \n", j);
            printf("input: ");
            for (int k=0; k<4; k++) {
                printf("%f ", input[k]);
            }puts("");
            printf("expect: ");
            for (int k=0; k<4; k++) {
                printf("%f ", output[k]);
            }
            puts("");
            net.PropagateLayers(input, output,false);
            printf("output: ");
            for (int k=0; k<4; k++) {
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

    srand(time(NULL));
    net.SetInputSize(128*128*3);    
    net.AppendLayer(new ConvLayer(128, 3, 1, 4, 9, 1, &sigmoid, 0.0005, 0.9, 1.0));
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
            int val = (int)(output[i*128 + j]*255/maxval);
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
    Identity id;    
    vector<double> input;
    vector<double> output;
    
    ConvLayer *cl = new ConvLayer(128, 3, 1, 4, 9, 1, &sigmoid, 0.0005, 0.9, 1.0);
    PoolLayer *pl = new PoolLayer(128, 1, 1, 3, &id, 1.0);
    
    srand(time(NULL));
    net.SetInputSize(128*128*3);        
    net.AppendLayer(cl);
    net.AppendLayer(pl);
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
  int MAX_FILE = 12500;
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
  Identity id;

  srand(time(NULL));
  net.SetInputSize(32*32*3);
  net.AppendLayer(new ConvLayer(32, 3, 1, 2, 5, 16, &rel, 0.01, 0.9, 1.0));
  net.AppendLayer(new PoolLayer(32, 16, 2, 2, &id, 1.0));
  net.AppendLayer(new ConvLayer(16, 16, 1, 2, 5, 32, &rel, 0.01, 0.9, 0.75));
  net.AppendLayer(new PoolLayer(16, 32, 2, 2, &id, 1.0));
  net.AppendLayer(new ConvLayer(8 , 32, 1, 2, 5, 64, &rel, 0.01, 0.9, 0.75));
  net.AppendLayer(new PoolLayer(8 , 64, 2, 2, &id, 1.0));
  net.AppendLayer(new FullyConnectedLayer(4*4*64, 64, &rel, 0.01, 0.9, 0.75));  
  net.AppendLayer(new FullyConnectedLayer(64, 2, &softmax, 0.01, 0.9, 0.5));
  net.ConnectLayers();
  

  FILE *logfp = fopen( "aclog" , "w" );
  fclose( logfp );
  
  int bloop = 0;
  while( ++bloop ){

    clock_t start = clock();
    
    cerr << bloop << endl;
    for( int loop = 0; loop < 100; loop++ ){
      in.clear();
      out.clear();

      int img_n = mt()%10000;
      
      if( mt() % 2 == 0 ){
	out.push_back( 1.0 ); out.push_back( 0.0 );
	sprintf( filename , "processed36/cat.%d.jpg" , img_n );
      } else {
	out.push_back( 0.0 ); out.push_back( 1.0 );
	sprintf( filename , "processed36/dog.%d.jpg" , img_n );
      }
      
      pixels = stbi_load( filename , &width , &height , &bpp , 0 );

      int py = mt()%4;
      int px = mt()%4;
      for( int k = 0; k < 3; k++ )
	for( int i = py; i < py+32; i++ )
	  for( int j = px; j < px+32; j++ )
	    in.push_back( (double)pixels[(i*width+j)*3+k] / 256.0 );

      stbi_image_free (pixels);      
    
      ins.clear();
      ins.push_back( in );
      outs.clear();
      outs.push_back( out );
    
      net.TrainNetwork(ins, outs);
      vector<double> out2(2);
    }

    namonakiacc = 0;
    for( int loop = 0; loop < 100; loop++ ){
      in.clear();
      out.clear(); out.resize(2);
      int ans;

      int img_n = 10000 + mt()%2500;      
      
      if( mt() % 2 == 0 ){
	ans = 0;
	sprintf( filename , "processed36/cat.%d.jpg" , img_n );
      } else {
	ans = 1;
	sprintf( filename , "processed36/dog.%d.jpg" , img_n );
      }
      
      pixels = stbi_load( filename , &width , &height , &bpp , 0 );

      int py = mt()%4;
      int px = mt()%4;
      for( int k = 0; k < 3; k++ )
	for( int i = py; i < py+32; i++ )
	  for( int j = px; j < px+32; j++ )
	    in.push_back( (double)pixels[(i*width+j)*3+k] / 256.0 );

      stbi_image_free (pixels);      

      net.PropagateLayers( in , out );

      int res = 0;
      if( out[0] < out[1] ) res = 1;

      if( res == ans ) namonakiacc++;
    }
    cerr << "ac : " << namonakiacc << " / 100" << endl;

    if( bloop == 1 && namonakiacc < 60 ) return;
    
    FILE *logfp = fopen( "aclog" , "a" );
    fprintf( logfp , "%d\n" , namonakiacc );
    fclose( logfp );

    FILE *fp = fopen( "stop_f" , "r" );
    int stop_f;
    fscanf( fp , "%d" , &stop_f );
    if( stop_f == 1 ) break;
    fclose( fp );

    clock_t end = clock();


    
    //cout << (double)(end - start) / CLOCKS_PER_SEC << "sec" << endl;

    if( bloop % 1000 == 0 ){
      fp = fopen( "submit.csv" , "w" );
      fprintf( fp , "id,label\n" );
      for( int loop = 1; loop <= 12500; loop++ ){
	cerr << loop << " / 12500" << endl;

	int catcnt = 0;
	int dogcnt = 0;

	sprintf( filename , "processedtest36/%d.jpg" , loop );
	
	pixels = stbi_load( filename , &width , &height , &bpp , 0 );

	
	for( int py = 0; py < 4; py++ ){
	  for( int px = 0; px < 4; px++ ){
	    in.clear();
	    out.clear(); out.resize(2);

	    for( int k = 0; k < 3; k++ )
	      for( int i = py; i < py+32; i++ )
		for( int j = px; j < px+32; j++ )
		  in.push_back( (double)pixels[(i*width+j)*3+k] / 256.0 );

	    net.PropagateLayers( in , out );

	    if( out[0] < out[1] ) dogcnt++;
	    else catcnt++;
	  }
	}

	stbi_image_free (pixels);

	int res = 0;
	if( catcnt < dogcnt ) res = 1;

	fprintf( fp , "%d,%d\n" , loop , res );
      }

      fclose( fp );
  

    }
  }

}


int rev( int x ){
  int res = 0;

  for( int i = 0; i < 4; i++ )
    res += ( (x>>(i*8)) & 255 ) << (24-i*8);

  return res;
}


double limg[60000][28*28+1];
int llabel[60000];
double timg[60000][28*28+1];
int tlabel[60000];


void TestMNIST(){

  mt19937 mt( time(NULL) );
  
  int magic_number;
  int N, Nt, Nl, H, W;
  
  vector<double> in, out;
  DoubleVector2d ins, outs;

  int namonakiacc = 0;

  NeuralNet net;
  RectifiedLinear rel;
  LogisticSigmoid sigmoid;
  Softmax softmax;
  Identity id;

  /*
  srand(time(NULL));
  net.SetInputSize(28*28);
  net.AppendLayer(new ConvLayer(28, 1, 1, 2, 5, 8, &sigmoid, 0.01, 0.9));
  net.AppendLayer(new PoolLayer(28, 8, 2, 2, &id));
  net.AppendLayer(new FullyConnectedLayer(14*14*8, 10, &softmax, 0.01, 0.9));
  net.ConnectLayers();
  */

  srand(time(NULL));
  net.SetInputSize(28*28);
  net.AppendLayer(new ConvLayer(28, 1, 1, 2, 5, 8, &rel, 0.01, 0.9, 1.0));
  net.AppendLayer(new PoolLayer(28, 8, 2, 2, &id, 1.0));
  net.AppendLayer(new ConvLayer(14, 8, 1, 2, 5, 16, &rel, 0.01, 0.9, 0.75));
  net.AppendLayer(new PoolLayer(14, 16, 3, 3, &id, 1.0));    
  net.AppendLayer(new FullyConnectedLayer(5*5*16, 10, &softmax, 0.01, 0.9, 0.5));
  net.ConnectLayers();

  /*
  FullyConnectedLayer *full1 = new FullyConnectedLayer(28*28, &sigmoid, 0.1);
  FullyConnectedLayer *full2 = new FullyConnectedLayer(32, &sigmoid, 0.1);  
  FullyConnectedLayer *full3 = new FullyConnectedLayer(10, &sigmoid, 0.1);
  
  srand(time(NULL));
  net.AppendLayer(full1);
  net.AppendLayer(full2);
  net.AppendLayer(full3);    
  net.ConnectLayers();  
  */

  FILE *fp;

  fp = fopen( "train-images-idx3-ubyte" , "rb" );
  
  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);
  Nl = N;

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
	limg[k][i*W+j] = (double)tmp / 256.0;
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
    llabel[k] = int(tmp);
  }

  fclose( fp );

  fp = fopen( "t10k-images-idx3-ubyte" , "rb" );
  
  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);
  Nt = N;

  fread( &H , sizeof( H ) , 1 , fp );
  H = rev(H);
  
  fread( &W , sizeof( W ) , 1 , fp );
  W = rev(W);


  for( int k = 0; k < N; k++ ){
    for( int i = 0; i < H; i++ ){
      for( int j = 0; j < W; j++ ){
	unsigned char tmp;
	fread( &tmp , sizeof( tmp ) , 1 , fp );
	timg[k][i*W+j] = (double)tmp / 256.0;
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
    tlabel[k] = int(tmp);
  }

  fclose( fp );
  
  int lloop = 0;
  int tloop = 0;
  int loop_n = 1000;

  for( int loop = 1; loop < loop_n; loop++ ){
    cerr << loop << " / " << loop_n << endl;    
    for( ; lloop < 100*loop; lloop++ ){
      in.clear();
      out = vector<double>(10,0.0);

      int lnum = mt()%Nl;

      for( int i = 0; i < H; i++ )
	for( int j = 0; j < W; j++ )
	  in.push_back( limg[lnum][i*W+j] );

      out[ llabel[lnum] ] = 1.0;

      ins.clear();
      ins.push_back( in );
      outs.clear();
      outs.push_back( out );

      net.TrainNetwork(ins, outs);
    }


    namonakiacc = 0;
    for( ; tloop < 1000*loop; tloop++ ){
      in.clear();
      out = vector<double>(10,0.0);

      int tnum = mt()%Nt;

      for( int i = 0; i < H; i++ )
	for( int j = 0; j < W; j++ )
	  in.push_back( timg[tnum][i*W+j] );

      net.PropagateLayers( in , out , false );

      int res = 0;
      for( int i = 1; i < 10; i++ )
	if( out[res] < out[i] ) res = i;

#if DEBUG
      printf( "out : %d, ans : %d\n" , res , tlabel[tloop%Nt] );
      for( int i = 0; i < 10; i++ )
	printf( "%lf " , out[i] );
      printf( "\n" );
#endif
      
      if( res == tlabel[tnum] ) namonakiacc++;
    }

    cerr << "ac : " << namonakiacc << " / " << 1000 << endl;

    if( loop == loop_n-1 ){
      int addc = 0;
      cerr << "addc : ";
      scanf( "%d" , &addc );
      loop_n += addc;
    }
  }

  namonakiacc = 0;
  for( tloop = 0; tloop < Nt; tloop++ ){
    in.clear();
    out = vector<double>(10,0.0);

    for( int i = 0; i < H; i++ )
      for( int j = 0; j < W; j++ )
	in.push_back( timg[tloop%Nt][i*W+j] );

    net.PropagateLayers( in , out );

    int res = 0;
    for( int i = 1; i < 10; i++ )
      if( out[res] < out[i] ) res = i;

    if( res == tlabel[tloop%Nt] ) namonakiacc++;
  }

  cout << "ac : " << namonakiacc << " / " << Nt << endl;
  
}


void TestMNISTwithsave(){

  mt19937 mt( time(NULL) );
  
  int magic_number;
  int N, Nt, Nl, H, W;
  
  vector<double> in, out;
  DoubleVector2d ins, outs;

  int namonakiacc = 0;

  bool recoglearn = true;

  NeuralNet net;
  RectifiedLinear rel;
  LogisticSigmoid sigmoid;
  Softmax softmax;
  Identity id;

  srand(time(NULL));
  if( recoglearn ){
    net.SetInputSize(28*28);
    net.AppendLayer(new ConvLayer(28, 1, 1, 2, 5, 8, &rel, 0.01, 0.9, 1.0));
    net.AppendLayer(new PoolLayer(28, 8, 2, 2, &id, 1.0));
    net.AppendLayer(new ConvLayer(14, 8, 1, 2, 5, 16, &rel, 0.01, 0.9, 0.75));
    net.AppendLayer(new PoolLayer(14, 16, 3, 3, &id, 1.0));
    net.AppendLayer(new FullyConnectedLayer(5*5*16, 10, &softmax, 0.01, 0.9, 0.5));
    net.ConnectLayers();
  } else {
    net.SetInputSize(28*28);
    net.AppendLayer(new ConvLayer(28, 1, 1, 2, 5, 8, &rel, 0.01, 0.9, 1.0));
    net.AppendLayer(new PoolLayer(28, 8, 2, 2, &id, 1.0));
    net.AppendLayer(new ConvLayer(14, 8, 1, 2, 5, 16, &rel, 0.01, 0.9, 0.75));
    net.AppendLayer(new PoolLayer(14, 16, 3, 3, &id, 1.0));
    net.AppendLayer(new FullyConnectedLayer(5*5*16, 28*28, &rel, 0.01, 0.9, 1.0));  
    net.ConnectLayers();

    net.SetLearningFlag( 0 , false );
    net.SetLearningFlag( 1 , false );
    net.SetLearningFlag( 2 , false );
    net.SetLearningFlag( 3 , false );
  }

  net.Load( "MNIST" );

  FILE *fp;

  fp = fopen( "train-images-idx3-ubyte" , "rb" );
  
  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);
  Nl = N;

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
	limg[k][i*W+j] = (double)tmp / 256.0;
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
    llabel[k] = int(tmp);
  }

  fclose( fp );

  fp = fopen( "t10k-images-idx3-ubyte" , "rb" );
  
  fread( &magic_number , sizeof( magic_number ) , 1 , fp );
  magic_number = rev( magic_number );

  fread( &N , sizeof( N ) , 1 , fp );
  N = rev(N);
  Nt = N;

  fread( &H , sizeof( H ) , 1 , fp );
  H = rev(H);
  
  fread( &W , sizeof( W ) , 1 , fp );
  W = rev(W);


  for( int k = 0; k < N; k++ ){
    for( int i = 0; i < H; i++ ){
      for( int j = 0; j < W; j++ ){
	unsigned char tmp;
	fread( &tmp , sizeof( tmp ) , 1 , fp );
	timg[k][i*W+j] = (double)tmp / 256.0;
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
    tlabel[k] = int(tmp);
  }

  fclose( fp );
  
  int loop = 0;
  
  while( ++loop ){
    cerr << loop << endl;

    // learn
    for( int lloop = 0; lloop < 100; lloop++ ){
      in.clear();
      out.clear();

      int lnum = mt()%Nl;

      for( int i = 0; i < H; i++ ){
	for( int j = 0; j < W; j++ ){
	  in.push_back( limg[lnum][i*W+j] );
	  if( !recoglearn ) out.push_back( limg[lnum][i*W+j] );	  
	}
      }

      if( recoglearn ){
	out.resize( 10 , 0.0 );
	out[ llabel[lnum] ] = 1.0;
      }

      ins.clear();
      ins.push_back( in );
      outs.clear();
      outs.push_back( out );

      net.TrainNetwork(ins, outs);
    }


    // recognize
    namonakiacc = 0;
    for( int tloop = 0; tloop < 100; tloop++ ){
      in.clear();
      if( recoglearn ) out = vector<double>(10,0.0);
      else out = vector<double>(28*28,0.0);

      int tnum = mt()%Nt;

      for( int i = 0; i < H; i++ )
	for( int j = 0; j < W; j++ )
	  in.push_back( timg[tnum][i*W+j] );

      net.PropagateLayers( in , out , false );

      // visualize
      if( tloop == 0 ){
	net.Visualize( loop , 0 , 28 , 0 );
	net.Visualize( loop , 2 , 14 , 0 );
	net.Visualize( loop , 4 , 7  , 0 );
	if( !recoglearn ) net.Visualize( loop , 5 , 28 , 0 );
      }

      if( recoglearn ){
	int res = 0;
	for( int i = 1; i < 10; i++ )
	  if( out[res] < out[i] ) res = i;

	if( res == tlabel[tnum] ) namonakiacc++;
      }
    }

    if( recoglearn ) cerr << "ac : " << namonakiacc << " / " << 100 << endl;

    fp = fopen( "stop_f" , "r" );
    int stop_f;
    fscanf( fp , "%d" , &stop_f );
    fclose( fp );
    if( stop_f == 1 ) break;
  }

  net.Save( "MNIST" );
}



void TestArtstyle(){

  mt19937 mt( time( NULL ) );
  int MAX_FILE = 12500;
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
  Identity id;

  srand(time(NULL));
  net.SetInputSize(128*128*3);
  net.AppendLayer(new ConvLayer(128, 3 , 1, 2, 5, 4, &rel, 0.01, 0.9, 1.0));
  net.AppendLayer(new PoolLayer(128, 4 , 2, 2, &id, 1.0));
  net.AppendLayer(new ConvLayer(64 , 4 , 1, 2, 5, 8, &rel, 0.01, 0.9, 0.75));
  net.AppendLayer(new PoolLayer(64 , 8 , 2, 2, &id, 1.0));
  net.AppendLayer(new ConvLayer(32 , 8 , 1, 2, 5, 16, &rel, 0.01, 0.9, 0.75));
  net.AppendLayer(new PoolLayer(32 , 16, 2, 2, &id, 1.0));
  net.AppendLayer(new FullyConnectedLayer(16*16*16, 64, &rel, 0.01, 0.9, 0.75));  
  net.AppendLayer(new FullyConnectedLayer(64, 2, &softmax, 0.01, 0.9, 0.5));
  net.ConnectLayers();

  net.Load( "dogandcat" );
  
  FILE *logfp = fopen( "aclog" , "w" );
  fclose( logfp );
  
  int bloop = 0;
  while( ++bloop ){
    
    cerr << bloop << endl;
    for( int loop = 0; loop < 100; loop++ ){
      in.clear();
      out.clear();

      int img_n = mt()%10000;
      
      if( mt() % 2 == 0 ){
	out.push_back( 1.0 ); out.push_back( 0.0 );
	sprintf( filename , "processed128/cat.%d.jpg" , img_n );
      } else {
	out.push_back( 0.0 ); out.push_back( 1.0 );
	sprintf( filename , "processed128/dog.%d.jpg" , img_n );
      }
      
      pixels = stbi_load( filename , &width , &height , &bpp , 0 );

      for( int k = 0; k < bpp; k++ )
	for( int i = 0; i < height; i++ )
	  for( int j = 0; j < width; j++ )
	    in.push_back( (double)pixels[(i*width+j)*3+k] / 256.0 );

      stbi_image_free (pixels);      
    
      ins.clear();
      ins.push_back( in );
      outs.clear();
      outs.push_back( out );
    
      net.TrainNetwork(ins, outs);
    }

    namonakiacc = 0;
    for( int loop = 0; loop < 100; loop++ ){
      in.clear();
      out.clear(); out.resize(2);
      int ans;

      int img_n = 10000 + mt()%2500;      
      
      if( mt() % 2 == 0 ){
	ans = 0;
	sprintf( filename , "processed128/cat.%d.jpg" , img_n );
      } else {
	ans = 1;
	sprintf( filename , "processed128/dog.%d.jpg" , img_n );
      }
      
      pixels = stbi_load( filename , &width , &height , &bpp , 0 );

      for( int k = 0; k < bpp; k++ )
	for( int i = 0; i < height; i++ )
	  for( int j = 0; j < width; j++ )
	    in.push_back( (double)pixels[(i*width+j)*3+k] / 256.0 );

      stbi_image_free (pixels);      

      net.PropagateLayers( in , out );

      // visualize
      if( loop == 0 ){
	net.Visualize( bloop , 0 , 128 , 0 );
	net.Visualize( bloop , 2 , 64 , 0 );
	net.Visualize( bloop , 4 , 32  , 0 );
      }

      
      int res = 0;
      if( out[0] < out[1] ) res = 1;

      if( res == ans ) namonakiacc++;
    }
    cerr << "ac : " << namonakiacc << " / 100" << endl;

    FILE *logfp = fopen( "aclog" , "a" );
    fprintf( logfp , "%d\n" , namonakiacc );
    fclose( logfp );

    FILE *fp = fopen( "stop_f" , "r" );
    int stop_f;
    fscanf( fp , "%d" , &stop_f );
    if( stop_f == 1 ) break;
    fclose( fp );
  }

  net.Save( "dogandcat" );  
}


int main(){

  //TestFullyConnectedLayer();
  //TestConvLayer();  
  //TestPoolLayer();
  //TestDeepLearning();
  //TestMNIST();
  //TestMNISTwithsave();
  TestArtstyle();  
}
