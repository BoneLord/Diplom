#include "layer.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
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
    srand(time(NULL));
    for (int i = 0; i < myPrevSize; ++i) {
        for (int j = 0; j < mySize; ++j) {
            myWeights[i][j] = 0.6 * rand() / RAND_MAX - 0.3;
        }
    }
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

double * Layer::computeOutput(double *input) const {
    double *inducedLocalFields = computeInducedLocalField(input);
    double *res = new double [mySize];
    for (int i = 0; i < mySize; ++i) {
        res[i] = activationFunction(inducedLocalFields[i]);
    }
    delete [] inducedLocalFields;
    return res;
}

double * Layer::computeInducedLocalField(double *input) const {
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

double * Layer::backPropagation(double *gradients) const {
    double *weightedSum = new double[myPrevSize];
    for (int i = 0; i < myPrevSize; ++i) {
        for (int j = 0; j < mySize; ++j) {
//            qDebug() << "gradients[j]" << gradients[j];
//            qDebug() << "myWeights[i][j]" << myWeights[i][j];
            weightedSum[i] += gradients[j] * myWeights[i][j];
//            qDebug() << "weightedSum[i]" << weightedSum[i];
        }
    }
    return weightedSum;
}

//NE PROVERENO!!!!
/*double** Layer::updateWeights(double **deltas, double regularizationParameter) {
    //double **oldWeights = copyWeights(mySize, myPrevSize);*/
void Layer::updateWeights(double **deltas, double regularizationParameter) {
    for (int j = 0; j < mySize; ++j) {
        myWeights[0][j] += deltas[0][j];
        for (int i = 1; i < myPrevSize; ++i) {
            myWeights[i][j] += deltas[i][j] + computeRegularization(regularizationParameter, myWeights[i][j]);
        }
    }
//    return oldWeights;
}

double Layer::computeRegularization(double regularizationParameter, double weight) const {
    double w0 = regularizationParameter * regularizationParameter;
    double w1 = weight * weight;
    return 2 * (w0 * weight) / ((w0 + w1) * (w0 + w1));
}

//double** Layer::updateWeights(double **deltas) {
//    return updateWeights(deltas, 0);
//}

void Layer::updateWeights(double **deltas) {
    updateWeights(deltas, 0);
}

double** Layer::copyWeights(int inputLength, int length) const {
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
