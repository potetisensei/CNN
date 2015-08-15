#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>
#include <vector>
#include <limits>

using namespace std;

typedef vector<vector<double> > DoubleVector2d;
typedef vector<DoubleVector2d> DoubleVector3d;

double GenRandom(double fmin, double fmax) {
    double f = (double)rand() / RAND_MAX;
    return fmin + f * (fmax - fmin);
}

class Weight {
public:
    Weight () {}
    double GetWeight() const { return *weight_instance_; }   
    void SetWeight(double weight) const { *weight_instance_ = weight; }
    void AddWeight(double weight) const { *weight_instance_ += weight; }
    void SubWeight(double weight) const { *weight_instance_ -= weight; }
    double *GetWeightInstance() { return weight_instance_; }
    void SetWeightInstance(double *ref) { weight_instance_ = ref; }

private:
    double *weight_instance_;
};

struct Edge {
    int to;
    Weight w;
};

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
    Layer(double rate) : calculated_(false), learning_rate_(rate) {}
    virtual ~Layer() {}

    virtual void ConnectLayer(Layer *layer) { assert(0); }
    virtual void CalculateOutput() { assert(0); }
    virtual void Propagate(Layer *output) { assert(0); }
    virtual void BackPropagate(DoubleVector2d next_deltas) { assert(0); }
    virtual void UpdateWeight(DoubleVector2d deltas) { assert(0); }
    virtual void UpdateBias(DoubleVector2d deltas) { assert(0); }

    bool calculated_;
    vector<struct Neuron> neurons_; // think as linear even 2d, 3d
    vector<vector<struct Edge> > edges_; // think as linear 
    DoubleVector2d deltas_; // [sample_idx][neuron_idx] 
protected:
    double learning_rate_;
};

/*class ConvLayer : public Layer {
};

class PoolLayer : public Layer {
};

class NormLayer : public Layer {
};*/

class FullConnectedLayer : public Layer {
public:
    FullConnectedLayer(int num_neurons, ActivationFunction *f, double learning_rate) : f_(f), Layer(learning_rate) {
        neurons_.resize(num_neurons);
        edges_.resize(num_neurons);
    }

    virtual ~FullConnectedLayer() {
        for (int i=0; i<neurons_.size(); i++) {
            for (int j=0; j<edges_[i].size(); j++) {
                delete edges_[i][j].w.GetWeightInstance();
            }
        }
    }   

    virtual void ConnectLayer(Layer *layer) {
        Weight weight;
        int num_output_neurons = layer->neurons_.size();
        
        for (int i=0; i<num_output_neurons; i++) {
            biases_.push_back(GenRandom(-0.5, 0.5));
        }

        for (int i=0; i<neurons_.size(); i++) {
            edges_[i].resize(num_output_neurons);
            for (int j=0; j<num_output_neurons; j++) {
                double *valp = new double(GenRandom(-1.0, 1.0));

                assert(-1.0 <= *valp && *valp <= 1.0);
                weight.SetWeightInstance(valp);
                edges_[i][j].to = j;
                edges_[i][j].w = weight;
            }
        }
    }

    virtual void CalculateOutput() {
        assert(!calculated_);
        for (int i=0; i<neurons_.size(); i++) {
            neurons_[i].z = f_->Calculate(neurons_[i].u, neurons_);
        }
        calculated_ = true;
    }

    virtual void Propagate(Layer *layer) {
        assert(calculated_);

        for (int i=0; i<layer->neurons_.size(); i++) {
            layer->neurons_[i].u = 0.0;
        }

        for (int i=0; i<neurons_.size(); i++) {
            for (int j=0; j<edges_[i].size(); j++) {
                struct Edge e = edges_[i][j];
                layer->neurons_[e.to].u += e.w.GetWeight() * neurons_[i].z;
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
                    double w = edges_[j][i].w.GetWeight();

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
        
                    edges_[i][k].w.SubWeight(learning_rate_ * deltas[l][j] * neurons_[i].z / deltas.size()); // Wji -= e * delta[j] * z[i] / N
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
    vector<double> biases_;
    ActivationFunction *f_;
};

class ConvNet {
public:
    ConvNet(/*ErrorFunction *e*/) : connected_(false)/*, e_(e)*/  {}
    ~ConvNet() {}

    void GetOutput(vector<double> &output) {
        int last_idx = layers_.size()-1;

        assert(last_idx >= 1);
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
            first_neurons[i].u = input[i];
        }

        layers_[0]->calculated_ = false;
        for (int i=0; i<layers_.size()-1; i++) {
            layers_[i]->CalculateOutput();
            layers_[i]->Propagate(layers_[i+1]);
        }
        layers_[last_idx]->CalculateOutput();

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

int main(void) {
    //SquaredError sqe = SquaredError();
    ConvNet net/*&sqe*/;
    RectifiedLinear rel;
    Softmax softmax;
    vector<double> input;
    vector<double> output;

    srand(time(NULL));
    printf("%f\n", rel.CalculateDerivative(0.5));
    net.AppendLayer(new FullConnectedLayer(3, &rel, 10e-4));
    net.AppendLayer(new FullConnectedLayer(2, &rel, 10e-4));
    net.AppendLayer(new FullConnectedLayer(3, &softmax, 10e-4));
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
