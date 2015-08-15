#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>
#include <vector>
#include <limits>
#include "bmp.h"
#include "layer.h"
#include "fully_connected_layer.h"

using namespace std;

typedef vector<vector<double> > DoubleVector2d;
typedef vector<DoubleVector2d> DoubleVector3d;
typedef vector<DoubleVector3d> DoubleVector4d;

double GenRandom(double fmin, double fmax) {
    double f = (double)rand() / RAND_MAX;
    return fmin + f * (fmax - fmin);
}

struct Neuron {
    double u;
    double z;
};

class ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> &neurons) {
        return -numeric_limits<double>::max();
    }

    virtual double CalculateDerivative(double u) {
        return -numeric_limits<double>::max();
    }
};

class LogisticSigmoid : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> &neurons) {
        return 1/(1 + exp(-u));
    }

    virtual double CalculateDerivative(double u) {
        double fu = 1/(1 + exp(-u));
        return fu * (1 - fu);
    }
};

class TangentSigmoid : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> &neurons) {
        return tanh(u);
    }

    virtual double CalculateDerivative(double u) {
        return (1 - tanh(u)*tanh(u));
    }
};

class RectifiedLinear : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> &neurons) {
        return max(u, 0.0);
    }

    virtual double CalculateDerivative(double u) {
        if (u >= 0) return 1.0;
        else return 0.0;
    }
};

class Identity : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> &neurons) {
        return u;
    }

    virtual double CalculateDerivative(double u) {
        return 1.0;
    }
};

class Softmax : public ActivationFunction {
public:
    virtual double Calculate(double u, vector<struct Neuron> &neurons) {
        double denominator = 0;

        for (int i=0; i<neurons.size(); i++) {
            denominator += exp(neurons[i].u);
        }
        return exp(u)/denominator;
    }

    virtual double CalculateDerivative(double u) {
        return -numeric_limits<double>::max();
    }
};


/*class ErrorFunction {
public:
    virtual double CalculateDerivative(double d, double y) {
        return -numeric_limits<double>::max();
    }

    virtual double CalculateOne(vector<double> &ds, vector<double> &ys) {
        assert(0);
    }

    double CalculateAll(DoubleVector2d &dataset, DoubleVector2d outputs) {
        double sum_all = 0.0;

        assert(dataset.size() == outputs.size());
        for (int i=0; i<dataset.size(); i++) {
            assert(dataset[i].size() == outputs[i].size());
            sum_all += CalculateOne(dataset[i], outputs[i]);
        }

        return sum_all / dataset.size();
    }
};
    
class SquaredError : ErrorFunction {
public:
    virtual double CalculateDerivative(double d, double y) {
        return d - y;
    }

    virtual double CalculateOne(vector<double> &ds, vector<double> &ys) {
        double sum_squared = 0.0;

        assert(ds.size() == ys.size());
        for (int i=0; i<ds.size(); i++) {
            double diff = ds[i]-ys[i];
            sum_squared += diff*diff;
        }
        return sum_squared/2;
    }
};

class CrossEntropy : ErrorFunction {
public:
    virtual double CalculateOne(vector<double> &ds, vector<double> &ys) {
        double sum_entropy = 0.0;

        assert(ds.size() == ys.size());
        for (int i=0; i<ds.size(); i++) {
            sum_entropy += ds[i] * log(ys[i]);
        }
        return -sum_entropy;
    }
};*/

class Layer {
public:
    Layer() : calculated_(false) {}
    Layer(double rate, ActivationFunction *f) 
        : calculated_(false), learning_rate_(rate), f_(f) {}
    virtual ~Layer() {}

    virtual void ConnectLayer(Layer *layer) { assert(0); }

    virtual void CalculateOutput(Layer *layer) {
        vector<struct Neuron> &neurons = layer->neurons_;

        assert(!layer->calculated_);
        for (int i=0; i<neurons.size(); i++) {
            neurons[i].z = f_->Calculate(neurons[i].u, neurons);
        }
        layer->calculated_ = true;
    }

    virtual void Propagate(Layer *layer) { assert(0); }
    virtual void BackPropagate(DoubleVector2d next_deltas) { assert(0); }
    virtual void UpdateWeight(DoubleVector2d deltas) { assert(0); }
    virtual void UpdateBias(DoubleVector2d deltas) { assert(0); }

    bool calculated_;
    vector<struct Neuron> neurons_; // think as 1d even if Layer has 2d or 3d neurons
    DoubleVector2d deltas_; // [sample_idx][neuron_idx] 
protected:
    ActivationFunction *f_;
    double learning_rate_;
};

class ConvLayer : public Layer {
public:
    ConvLayer(int breadth_neuron, int num_channels, int stride, int breadth_filter, int num_filters, ActivationFunction *f, double learning_rate)
            :  breadth_neuron_(breadth_neuron), 
               num_channels_(num_channels), 
               stride_(stride), 
               breadth_filter_(breadth_filter),
               num_filters_(num_filters),
               Layer(learning_rate, f) {
        assert(stride >= 1);
        neurons_.resize(breadth_neuron * breadth_neuron * num_channels); // dangerous
        edges_weight_.resize(num_filters);
        breadth_output_ = (breadth_neuron-1)/stride + 1;
    }

    virtual ~ConvLayer() {}

    virtual void ConnectLayer(Layer *layer) {
        int expected_size = num_filters_ * breadth_output_ * breadth_output_; // dangerous

        assert(layer->neurons_.size() == expected_size);

        for (int i=0; i<num_filters_; i++) { // Bijm == Bm
            biases_.push_back(GenRandom(-0.5, 0.5));
        }

        double lim = 1.0/sqrt(breadth_filter_*breadth_filter_*num_channels_);
        assert(edges_weight_.size() == num_filters_);
        for (int m=0; m<num_filters_; m++) {
            edges_weight_[m].resize(num_channels_);
            for (int k=0; k<num_channels_; k++) {
                edges_weight_[m][k].resize(breadth_filter_);
                for (int i=0; i<breadth_filter_; i++) {
                    edges_weight_[m][k][i].resize(breadth_filter_);
                    for (int j=0; j<breadth_filter_; j++) {
                        edges_weight_[m][k][i][j] = GenRandom(-lim, lim);
                    }
                }
            }
        }
    }

 
    //    3d_neurons[z][y][x] z := m, k, z...
    //                        y := i, p, y...
    //                        x := j, q, x...
    virtual void Propagate(Layer *layer) {
        vector<struct Neuron> &output_neurons = layer->neurons_;
        int size = breadth_output_ * breadth_output_;

        assert(calculated_);

        for (int i=0; i<output_neurons.size(); i++) {
            output_neurons[i].u = 0.0;
        }

        assert(biases_.size() == num_filters_);
        for (int m=0; m<num_filters_; m++) { 
            for (int i=0; i<breadth_output_; i++) {
                for (int j=0; j<breadth_output_; j++) {
                    int neuron_idx1 = m*size + i*breadth_output_ + j;
                    double sum_conv = 0.0;

                    assert(i*stride_ < breadth_neuron_);
                    assert(j*stride_ < breadth_neuron_);

                    for (int k=0; k<num_channels_; k++) {
                        for (int p=0; p<breadth_filter_; p++) {
                            for (int q=0; q<breadth_filter_; q++) {
                                int x = j*stride_ + q;
                                int y = i*stride_ + p;
                                int neuron_idx2 = k*size + y*breadth_output_ + x;
                                double z = 0.0;

                                if (x < breadth_neuron_ && y < breadth_neuron_) {
                                    z = neurons_[neuron_idx2].z;
                                }
                                sum_conv += z * edges_weight_[m][k][p][q];
                            }
                        }
                    }

                    output_neurons[neuron_idx1].u = sum_conv + biases_[m];
                }
            }
        }

        layer->calculated_ = false;
    }

    virtual void BackPropagate(DoubleVector2d next_deltas) { assert(0); }
    virtual void UpdateWeight(DoubleVector2d deltas) { assert(0); }
    virtual void UpdateBias(DoubleVector2d deltas) { assert(0); }

private:
    int breadth_neuron_;
    int num_channels_;
    int stride_;
    int breadth_filter_;
    int num_filters_;
    int breadth_output_;
    vector<double> biases_;
public:
    DoubleVector4d edges_weight_; // [filter_idx][channel_idx][y][x] weight-sharing
};

class PoolLayer : public Layer {
    PoolLayer(int breadth_neuron, int num_channels, int stride, int breadth_filter)
        : breadth_neuron_(breadth_neuron), 
          num_neurons_(num_neurons),
          stride_(stride),
          breadth_filter_(breadth_filter) {
              
    }

    virtual ~PoolLayer() {}

    virtual void ConnectLayer(Layer *layer) { assert(0); }

    virtual void CalculateOutput(Layer *layer) {
        vector<struct Neuron> &neurons = layer->neurons_;

        assert(!layer->calculated_);
        for (int i=0; i<neurons.size(); i++) {
            neurons[i].z = f_->Calculate(neurons[i].u, neurons);
        }
        layer->calculated_ = true;
    }

    virtual void Propagate(Layer *layer) { assert(0); }
    virtual void BackPropagate(DoubleVector2d next_deltas) { assert(0); }
    virtual void UpdateWeight(DoubleVector2d deltas) { assert(0); }
    virtual void UpdateBias(DoubleVector2d deltas) { assert(0); }

    bool calculated_;
    vector<struct Neuron> neurons_; // think as 1d even if Layer has 2d or 3d neurons
    DoubleVector2d deltas_; // [sample_idx][neuron_idx] 
protected:
    ActivationFunction *f_;
    double learning_rate_;
};

/*class NormLayer : public Layer {
};*/

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(int num_neurons, ActivationFunction *f, double learning_rate) : Layer(learning_rate, f) {
        neurons_.resize(num_neurons);
        edges_.resize(num_neurons);
    }

    virtual ~FullyConnectedLayer() {
        /*for (int i=0; i<neurons_.size(); i++) {
            for (int j=0; j<edges_[i].size(); j++) {
                delete edges_[i][j].w.GetWeightInstance();
            }
        }*/
    }   

    virtual void ConnectLayer(Layer *layer) {
        //Weight weight;
        int num_output_neurons = layer->neurons_.size();
        
        for (int i=0; i<num_output_neurons; i++) {
            biases_.push_back(GenRandom(-0.5, 0.5));
        }

        for (int i=0; i<neurons_.size(); i++) {
            edges_[i].resize(num_output_neurons);
            for (int j=0; j<num_output_neurons; j++) {
                //double *valp = new double(GenRandom(-1.0, 1.0));

                //assert(-1.0 <= *valp && *valp <= 1.0);
                //weight.SetWeightInstance(valp);
                edges_[i][j].to = j;
                edges_[i][j].w = GenRandom(-1.0, 1.0);//weight;
            }
        }
    }

    virtual void Propagate(Layer *layer) {
        assert(calculated_);

        for (int i=0; i<layer->neurons_.size(); i++) {
            layer->neurons_[i].u = 0.0;
        }

        for (int i=0; i<neurons_.size(); i++) {
            for (int j=0; j<edges_[i].size(); j++) {
                struct Edge e = edges_[i][j];
                layer->neurons_[e.to].u += e.w/*.GetWeight()*/ * neurons_[i].z;
            }
        }

        assert(biases_.size() == layer->neurons_.size());
        for (int i=0; i<biases_.size(); i++) {
            layer->neurons_[i].u += biases_[i];
        }

        layer->calculated_ = false;
    }

    virtual void BackPropagate(DoubleVector2d next_deltas) {
        deltas_.resize(next_deltas.size());
        for (int i=0; i<deltas_.size(); i++) {
            vector<double> &next_delta = next_deltas[i];
            vector<double> &delta = deltas_[i];
            
            delta.resize(neurons_.size());
            fill(delta.begin(), delta.end(), 0.0);

            for (int j=0; j<edges_.size(); j++) {
                for (int i=0; i<edges_[j].size(); i++) {
                    int k = edges_[j][i].to;
                    double w = edges_[j][i].w/*.GetWeight()*/;

                    delta[j] += next_delta[k] * w * f_->CalculateDerivative(neurons_[j].u);
                }
            }
        }
    }

    virtual void UpdateWeight(DoubleVector2d deltas) {
        for (int l=0; l<deltas.size(); l++) {
            assert(neurons_.size() == edges_.size());
            for (int i=0; i<edges_.size(); i++) {
                for (int k=0; k<edges_[i].size(); k++) {
                    int j = edges_[i][k].to;
        
                    edges_[i][k].w/*.SubWeight(*/-= learning_rate_ * deltas[l][j] * neurons_[i].z / deltas.size()/*)*/; // Wji -= e * delta[j] * z[i] / N
                }
            }
        }
    }

    virtual void UpdateBias(DoubleVector2d deltas) {
        for (int l=0; l<deltas.size(); l++) {
            assert(deltas[l].size() == biases_.size());
            for (int i=0; i<deltas[l].size(); i++) {
                biases_[i] -= learning_rate_ * deltas[l][i] / deltas.size(); // bi -= e * delta[i] / N
            }
        }
    }

private:
    EdgeVector2d edges_;
    vector<double> biases_;
};

class ConvNet {
public:
    ConvNet(/*ErrorFunction *e*/) : connected_(false)/*, e_(e)*/  {}
    ~ConvNet() {}

    void GetOutput(vector<double> &output) {
        int last_idx = layers_.size()-1;

        assert(last_idx >= 1);
        assert(layers_[last_idx]->calculated_);
        vector<struct Neuron> &neurons = layers_[last_idx]->neurons_;
        assert(neurons.size() == output.size());

        for (int i=0; i<neurons.size(); i++) {
            output[i] = neurons[i].z;
        }
    }

    void AppendLayer(Layer *layer) {
        assert(!connected_);
        layers_.push_back(layer);
    }

    void ConnectLayers() {
        assert(!connected_);   
        for (int i=0; i<layers_.size()-1; i++) {
            layers_[i]->ConnectLayer(layers_[i+1]);
        }
        connected_ = true;
    }

    void PropagateLayers(vector<double> &input, vector<double> &output) {
        int last_idx = layers_.size()-1;

        assert(connected_);
        assert(last_idx >= 1);

        vector<struct Neuron> &first_neurons = layers_[0]->neurons_;
        vector<struct Neuron> &last_neurons = layers_[last_idx]->neurons_;
        assert(first_neurons.size() == input.size());
        assert(last_neurons.size() == output.size());

        for (int i=0; i<first_neurons.size(); i++) {
            first_neurons[i].z = input[i];
        }

        layers_[0]->calculated_ = true;
        for (int i=0; i<=last_idx-1; i++) {
            layers_[i]->Propagate(layers_[i+1]);
            layers_[i]->CalculateOutput(layers_[i+1]);
        }

        assert(layers_[last_idx]->calculated_);
        for (int i=0; i<last_neurons.size(); i++) {
            output[i] = last_neurons[i].z;
        }
    }

    void BackPropagateLayers(DoubleVector2d &dataset, DoubleVector2d &outputs) {
        int last_idx = layers_.size()-1;

        assert(last_idx >= 1);
        assert(dataset.size() == outputs.size());
        DoubleVector2d &deltas = layers_[last_idx]->deltas_;

        deltas.resize(dataset.size());
        for (int i=0; i<dataset.size(); i++) {
            assert(dataset[i].size() == outputs[i].size());
            deltas[i].resize(dataset[i].size());
            for (int j=0; j<dataset[i].size(); j++) {
                deltas[i][j] = outputs[i][j] - dataset[i][j];
            }
        }

        layers_[last_idx-1]->UpdateWeight(deltas);
        layers_[last_idx-1]->UpdateBias(deltas);
        for (int i=last_idx-1; i>=1; i--) {
             layers_[i]->BackPropagate(layers_[i+1]->deltas_);
             layers_[i-1]->UpdateWeight(layers_[i]->deltas_);
             layers_[i-1]->UpdateBias(layers_[i]->deltas_);
        }
    }
    
    void TrainNetwork(DoubleVector2d &inputs, DoubleVector2d &expected_outputs) {
        DoubleVector2d actual_outputs;

        assert(inputs.size() == expected_outputs.size());
        actual_outputs.resize(inputs.size());
        for (int i=0; i<inputs.size(); i++) {
            assert(inputs[i].size() == expected_outputs[i].size());
            actual_outputs[i].resize(inputs[i].size());
            PropagateLayers(inputs[i], actual_outputs[i]);
        }

        BackPropagateLayers(expected_outputs, actual_outputs);
    }

private:
    bool connected_;
    vector<Layer *> layers_;
    //ErrorFunction *e_;
};

void TestFullyConnectedLayer() {
    //SquaredError sqe = SquaredError();
    ConvNet net/*&sqe*/;
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
    ConvNet net/*&sqe*/;
    RectifiedLinear rel;
    LogisticSigmoid sigmoid;
    Softmax softmax;
    vector<double> input;
    vector<double> output;
    ConvLayer *cl = new ConvLayer(128, 3, 1, 9, 1, &sigmoid, 0.0005);

    srand(time(NULL));
    printf("%f\n", rel.CalculateDerivative(0.5));
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


int main() {
    //TestFullyConnectedLayer();
    TestConvLayer();
}

