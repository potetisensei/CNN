#ifndef _UTIL_TYPE_H_
#define _UTIL_TYPE_H_

#include <cstdlib>
#include <vector>
using namespace std;

typedef vector<vector<double> > DoubleVector2d;
typedef vector<DoubleVector2d> DoubleVector3d;
typedef vector<DoubleVector3d> DoubleVector4d;

struct Neuron {
    double u;
    double z;
};

struct Weight {
    double val; // weight
    double lazy_sub; // sum of dEn/dWij in a mini-batch datasets
    int count; // number of mini-batch dataset
};

typedef vector<vector<struct Weight> > WeightVector2d;

inline double GenRandom(double fmin, double fmax) {
    double f = (double)rand() / RAND_MAX;
    return fmin + f * (fmax - fmin);
}

#endif
