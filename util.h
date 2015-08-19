#ifndef _UTIL_TYPE_H_
#define _UTIL_TYPE_H_

#include <cstdlib>
#include <vector>
#include <random>
using namespace std;

typedef vector<vector<double> > DoubleVector2d;
typedef vector<DoubleVector2d> DoubleVector3d;
typedef vector<DoubleVector3d> DoubleVector4d;

struct Neuron {
    double u;
    double z;
};

typedef vector<vector<struct Neuron> > NeuronVector2d;
typedef vector<NeuronVector2d> NeuronVector3d;
typedef vector<NeuronVector3d> NeuronVector4d;


struct Weight {
  double val; // weight
  double lazy_sub; // sum of dEn/dWij in a mini-batch datasets
  double prev_delta;
  int count; // number of mini-batch dataset
};

typedef vector<vector<struct Weight> > WeightVector2d;
typedef vector<WeightVector2d> WeightVector3d;
typedef vector<WeightVector3d> WeightVector4d;

extern mt19937 mt;
extern uniform_real_distribution<> urand;

inline double GenRandom(double fmin, double fmax) {
  //double f = urand(mt);
  double f = (double)rand() / RAND_MAX;
  return fmin + f * (fmax - fmin);
}

#endif
