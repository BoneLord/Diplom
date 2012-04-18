#ifndef LAYER_H
#define LAYER_H

class Layer {
private:
    double **myWeights;
    int myPrevSize;
    int mySize;
    void initialization();
    Layer& operator=(const Layer& another);//{ return *this; }
    Layer(const Layer& another);//{}
    double * computeInducedLocalField(double *input);
    double activationFunction(double outputLayer) const;
    double computeRegularization(double regularizationParameter, double weight);
    double** copyWeights(int inputLength, int length);
public:
    Layer(int prevLayerSize, int layerSize);
    ~Layer();
    int getPrevSize() const;
    int getSize() const;
    int getDimension() const;
    //double activationFunction(double outputLayer) const;
    double getWeights(int prevLayerSize, int layerSize) const;
    void setWeights(int prevLayerSize, int layerSize, double weigth);
    double * computeOutput(double *input);
    double * backPropagation(double *gradients);
    double** updateWeights(double **deltas, double regularizationParameter);
    double** updateWeights(double **deltas);
};

#endif // LAYER_H
