#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include <vector>
#include "trainingsetelement.h"

class NeuralNetwork {
private:
    friend class Trainer;
    class Trainer {
    private:
        NeuralNetwork *myNet;
        std::vector<double*> **myTrainingSet;
        int myMaxClusterSize;
        std::vector<double> myErrors;
        double **myRightAnswers;
        int myRightAnswersLength;
        int myTrainingSetSize;
        double ***myPreviousDeltas;
        double myMomentConstant;
        double myRateOfLearning;
        double myRegularizationParameter;
        void backwardComputation(double **layersOutputs, double *rightAnswer);
        double** computeGradients(double **layersOutputs, double *rightAnswer);
        double* computeGradientForOutputLayer(double *layerOutput, double *rightAnswer);
        double* computeGradientForHiddenLayer(Layer *layer, double *weightedSum, double *layerOutput);
        void updateWeights(double **gradients, double **layersOutputs);
        double** computeDeltas(double *gradient, int length, double *layerInput, int inputLength, double **previousDelta);
        double getPreviousDelta(double **previousDelta, int i, int j);
        double computeError(double *rightAnswer, double *networkOutput);
        double epoch();
        void shuffleTrainingSet();
        void fillTrainingSet(TrainingSetElement **trainingSetElements, int setSize, int outputVectorDimension);
    public:
        void train();
        Trainer(NeuralNetwork *net, TrainingSetElement **trainingSetElements,
                double rateOfLearning, double regularizationParameter, int setSize);
        ~Trainer();
    };
    int myLayerCount;
    Layer **myLayer;
    double myMinInput;
    double myMaxInput;
    double ** forwardComputation(double *vector);
    double * normalize(double *input);
    Layer* getOutputLayer() const;
    NeuralNetwork(const NeuralNetwork& another);
    NeuralNetwork& operator=(const NeuralNetwork& another);
public:
    NeuralNetwork(int layerCount, int *layerSizes, double minInput, double maxInput);
    ~NeuralNetwork();
    int recognize(double * input);    
    double * advancedRecognize(double *input);
    void train(TrainingSetElement **input, int setSize, double rateOfLearning, double regularizationParameter);

    void setTestWeights();
    void TestBackPropagation();
};

#endif // NEURALNETWORK_H
