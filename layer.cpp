#include "layer.h"
#include <QTime>
#include <math.h>
#include <QDebug>

Layer::Layer(int prevLayerSize, int layerSize) {
    myPrevSize = prevLayerSize+1;
    mySize = layerSize;
    myWeights = new double * [myPrevSize];
    for (int i = 0; i < myPrevSize; ++i) {
        myWeights[i] = new double [mySize];
    }
    initialization();
}

Layer::~Layer() {
    for (int i = 0; i < myPrevSize; ++i) {
        delete [] myWeights[i];
    }
    delete [] myWeights;
}

void Layer::initialization() {
    qsrand(QTime::currentTime().msec());
    for (int i = 0; i < myPrevSize; ++i) {
        for (int j = 0; j < mySize; ++j) {
            myWeights[i][j] = 0.6 * (qrand() % 2) - 0.3;
        }
    }
}

int Layer::getDimension() const {
    return mySize;
}

int Layer::getPrevSize() const {
    return myPrevSize;
}

int Layer::getSize() const {
    return mySize;
}

double Layer::activationFunction(double outputLayer) const {
    return 1 / (1 + exp(-outputLayer));
}

double Layer::getWeights(int prevLayerSize, int layerSize) const {
    return myWeights[prevLayerSize][layerSize];
}

void Layer::setWeights(int prevLayerSize, int layerSize, double weigth) {
    myWeights[prevLayerSize][layerSize] = weigth;
}

double * Layer::computeOutput(double *input) {
    double *inducedLocalFields = computeInducedLocalField(input);
    double *res = new double [mySize];
    for (int i = 0; i < mySize; ++i) {
        res[i] = activationFunction(inducedLocalFields[i]);
    }
    delete [] inducedLocalFields;
    return res;
}

double * Layer::computeInducedLocalField(double *input) {
    double *inducedLocalFields = new double [mySize];
    for (int i = 0; i < mySize; ++i) {
        double inducedLocalField = myWeights[0][i];
        for (int j = 0; j < myPrevSize - 1; ++j) {
            inducedLocalField += input[j] * myWeights[j+1][i];
        }
        inducedLocalFields[i] = inducedLocalField;
    }
    return inducedLocalFields;
}

double * Layer::backPropagation(double *gradients) {
    double *weightedSum = new double[myPrevSize];
    for (int i = 0; i < myPrevSize; ++i) {
        for (int j = 0; j < mySize; ++j) {
            weightedSum[i] += gradients[j] * myWeights[i][j];
        }
    }
    return weightedSum;
}

//NE PROVERENO!!!!
double** Layer::updateWeights(double **deltas, double regularizationParameter) {
    double **oldWeights = copyWeights(mySize, myPrevSize);
    for (int i = 0; i < myPrevSize; ++i) {
        double *weights = myWeights[i];
        double *delta = deltas[i];

        int l = mySize - 1;
        for (int j = 0; j < l; ++j) {
            weights[j] += delta[j] + computeRegularization(regularizationParameter, weights[j]);
        }
        weights[l] += delta[l];
    }
    return oldWeights;
}

double Layer::computeRegularization(double regularizationParameter, double weight) {
    double w0_2 = pow(regularizationParameter,2);
    return 2 * (w0_2 * weight) / pow(w0_2 + pow(weight,2),2);
}

double** Layer::updateWeights(double **deltas) {
    return updateWeights(deltas, 0);
}

double** Layer::copyWeights(int inputLength, int length) {
    double **oldWeights = new double * [length];
    for (int i = 0; i < length; ++i) {
        oldWeights[i] = new double [inputLength];
    }
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j < inputLength; ++j) {
            oldWeights[i][j] = myWeights[i][j];
        }
    }
    return oldWeights;
}
