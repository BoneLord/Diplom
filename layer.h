#ifndef LAYER_H
#define LAYER_H

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <QDebug>

class Layer {
private:
    double **myWeights;
    int myPrevSize;
    int mySize;
    void initialization();
    Layer& operator=(const Layer& another);
    Layer(const Layer& another);
    double * computeInducedLocalField(double *input) const;
    double activationFunction(double outputLayer) const;
    double computeRegularization(double regularizationParameter, double weight) const;
    double** copyWeights(int inputLength, int length) const;
public:
    Layer(int prevLayerSize, int layerSize);
    ~Layer();
    int getPrevSize() const;
    int getSize() const;
    double * computeOutput(double *input) const;
    double * backPropagation(double *gradients) const;

    const double* getNeuronWeights(int numberNeuron) const;
    void setWeightsLayer(double **weights);
    double getWeights(int prevLayerSize, int layerSize) const; // Test method
    void setWeights(int prevLayerSize, int layerSize, double weigth); // Test method

    void updateWeights(double **deltas, double regularizationParameter);
    void updateWeights(double **deltas);
//    double** updateWeights(double **deltas, double regularizationParameter);
//    double** updateWeights(double **deltas);
};

#endif // LAYER_H
