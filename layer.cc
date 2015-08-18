#include "layer.h"

void Layer::CalculateOutputUnits(vector<struct Neuron> units) {
    for (int i=0; i<units.size(); i++) {
        units[i].z = f_->Calculate(units[i].u, units);
    }
}
