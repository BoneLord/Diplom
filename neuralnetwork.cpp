#include "neuralnetwork.h"
#include <QDebug>
#include <float.h>
#include <math.h>
#include <limits.h>

NeuralNetwork::NeuralNetwork(int layerCount, int *layerSizes, double minInput, double maxInput) {
    myLayerCount = layerCount-1;
    myMinInput = minInput;
    myMaxInput = maxInput;
    myLayer = new Layer * [myLayerCount];
    for (int i = 0; i < myLayerCount; ++i) {
        myLayer[i] = new Layer(layerSizes[i], layerSizes[i+1]);
     }
}

NeuralNetwork::~NeuralNetwork() {
    for (int i = 0; i < myLayerCount; ++i) {
        delete myLayer[i];
    }
    delete [] myLayer;
}

int NeuralNetwork::recognize(double * input) {
    double * prevOutput = advancedRecognize(input);
    double max = DBL_MIN;
    int maxIndex = -1;
    int length = myLayer[myLayerCount-1]->getSize();
    for (int i = 0; i < length; ++i) {
        double value = prevOutput[i];
        qDebug() << "Value in neuron exit = " << value;
        if (value > max) {
            max = value;
            maxIndex = i;
        }
    }
    delete [] prevOutput;
    return maxIndex;
 }

double * NeuralNetwork::advancedRecognize(double *input) {
    double *normInput = normalize(input);
    double **layersOutputs = forwardComputation(normInput);
    double *lastLayerOutput = layersOutputs[myLayerCount];
    for (int i = 0; i < myLayerCount; ++i) {
        delete [] layersOutputs[i];
    }
    delete [] layersOutputs;
    return lastLayerOutput;
//    return forwardComputation(normalize(input))[myLayerCount];
}

double ** NeuralNetwork::forwardComputation(double *vector) {
    int length = myLayerCount;
    double **layersOutputs = new double * [length + 1];
    layersOutputs[0] = vector;
    for (int i = 0; i < length; ++i) {
        layersOutputs[i + 1] = myLayer[i]->computeOutput(layersOutputs[i]);
    }
    return layersOutputs;
}

double * NeuralNetwork::normalize(double *input) {
    double fr = (1 - 0) / 10;
    double functionMin = 0 + fr;
    double functionMax = 1 - fr;
    int length = myLayer[0]->getPrevSize() - 1;
    double * res = new double[length];
    double t = (functionMax - functionMin) / (myMaxInput - myMinInput);
    for (int i = 0; i < length; ++i) {
        double v = input[i];
        res[i] = t * v + (functionMax - t * myMaxInput);        
    }
    return res;
}

Layer* NeuralNetwork::getOutputLayer() const {
    return myLayer[myLayerCount-1];
}

void NeuralNetwork::train(TrainingSetElement **input, int setSize, double rateOfLearning, double regularizationParameter) {
//     NeuralNetwork::Trainer *trainer;
//     trainer = new NeuralNetwork::Trainer::Trainer(this, input, rateOfLearning, regularizationParameter, setSize);
//     trainer->train();
    NeuralNetwork::Trainer trainer(this, input, rateOfLearning, regularizationParameter, setSize);
    trainer.train();
}

void NeuralNetwork::setTestWeights() {
    myLayer[0]->setWeights(0,0,0.1);
    myLayer[0]->setWeights(0,1,0.1);
    myLayer[0]->setWeights(1,0,-0.2);
    myLayer[0]->setWeights(1,1,-0.1);
    myLayer[0]->setWeights(2,0,0.1);
    myLayer[0]->setWeights(2,1,0.3);
    myLayer[1]->setWeights(0,0,0.2);
    myLayer[1]->setWeights(1,0,0.2);
    myLayer[1]->setWeights(2,0,0.3);
//    myLayer[0]->setWeights(0,0,0.10075);
//    myLayer[0]->setWeights(0,1,0.10125);
//    myLayer[0]->setWeights(1,0,-0.199925);
//    myLayer[0]->setWeights(1,1,-0.099875);
//    myLayer[0]->setWeights(2,0,0.100675);
//    myLayer[0]->setWeights(2,1,0.301125);
//    myLayer[1]->setWeights(0,0,0.217);
//    myLayer[1]->setWeights(1,0,0.209);
//    myLayer[1]->setWeights(2,0,0.31);
//    for (int layerNum = 0; layerNum < myLayerCount; ++layerNum) {
//        for (int i = 0; i < myLayer[layerNum]->getPrevSize(); ++i) {
//            for (int j = 0; j < myLayer[layerNum]->getSize(); ++j) {
//                qDebug() << myLayer[layerNum]->getWeights(i,j);
//            }
//        }
//    }    
}

void NeuralNetwork::outputWeights() {
    for (int layerNum = 0; layerNum < myLayerCount; ++layerNum) {
        for (int i = 0; i < myLayer[layerNum]->getPrevSize(); ++i) {
            for (int j = 0; j < myLayer[layerNum]->getSize(); ++j) {
                qDebug() << myLayer[layerNum]->getWeights(i,j);
            }
        }
    }
}

//==========================================================================================================================
// Class Trainer
//==========================================================================================================================

NeuralNetwork::Trainer::Trainer(NeuralNetwork *net,TrainingSetElement **trainingSetElements,
                                double rateOfLearning, double regularizationParameter, int setSize) {
    myRateOfLearning = rateOfLearning;
    myRegularizationParameter = regularizationParameter;

    myPreviousDeltas = new double ** [net->myLayerCount];
    myMomentConstant = 0.3;
    Layer* outputLayer = net->getOutputLayer();
    myNet = net;
    int outputVectorDimension = outputLayer->getSize();
    myRightAnswersLength = outputVectorDimension;

    myTrainingSet = new std::vector<double*>* [outputVectorDimension];
    fillTrainingSet(trainingSetElements, setSize, outputVectorDimension);

    int trainingSetSize = 0;
    int max = INT_MIN;

    for (int i = 0; i < outputVectorDimension; ++i) {
        int size = myTrainingSet[i]->size();
//        if (size == 0) {
//            throw new NetworkTrainingException("Invalid training set");
//        }
        if (size > max) {
            max = size;
        }
        trainingSetSize += size;
    }

    myMaxClusterSize = max;
    myTrainingSetSize = trainingSetSize;

    myRightAnswers = new double * [outputVectorDimension];
    for (int i = 0; i < outputVectorDimension; ++i) {
        myRightAnswers[i] = new double [outputVectorDimension];
    }

    double functionMax = 0.9;
    double functionMin = 0.;
    for (int i = 0; i < outputVectorDimension; ++i) {
        double *rightAnswer = myRightAnswers[i];
        for (int j = 0; j < outputVectorDimension; ++j) {
            rightAnswer[j] = functionMin;
        }
        rightAnswer[i] = functionMax;
    }
}

NeuralNetwork::Trainer::~Trainer() {
    for (int i = 0; i < myRightAnswersLength; ++i) {
        delete [] myRightAnswers[i];
        delete myTrainingSet[i];
    }
    delete [] myRightAnswers;
    delete [] myTrainingSet;

    int length = myNet->myLayerCount;
    for (int i = 0; i < length; ++i) {
        int len = myNet->myLayer[i]->getPrevSize();
        for (int j = 0; j < len; ++j) {
            delete [] myPreviousDeltas[i][j];
        }
        delete [] myPreviousDeltas[i];
    }
    delete [] myPreviousDeltas;
}

void NeuralNetwork::Trainer::fillTrainingSet(TrainingSetElement **trainingSetElements,
                                             int setSize, int outputVectorDimension) {
    for (int i = 0; i < outputVectorDimension; ++i) {
        myTrainingSet[i] = new std::vector<double*>;
    }
    for (int i = 0; i < setSize; ++i) {
        TrainingSetElement *element = trainingSetElements[i];
        int index = element->getRightAnswer();
        myTrainingSet[index]->push_back(myNet->normalize(element->getData()));
    }
}

void NeuralNetwork::Trainer::train() {
    double initialRateLearning = myRateOfLearning;
    double prevError;
    double error = DBL_MAX;
    int count = 0;
    double deltaError;
    do {
        prevError = error;
        error = epoch();
        ++count;
        myErrors.push_back(error);
        deltaError = fabs(prevError - error);
        myRateOfLearning = initialRateLearning / (1 + count / 10.);
    }
    while (deltaError > 0.01 || deltaError / prevError > 0.001 || count > 500);
}

double NeuralNetwork::Trainer::epoch() {
//    shuffleTrainingSet();

    double averageError = 0;
    int length = myRightAnswersLength;
    for (int i = 0; i < myMaxClusterSize; ++i) {
        if (myTrainingSet[length - 1]->size() == i) {
            --length;
        }
        for (int j = 0; j < length; ++j) {
            double *rightAnswer = myRightAnswers[j];
            std::vector<double*> *vector = myTrainingSet[j];
            double *inputVector = vector->at(i);
            double **layersOutputs = myNet->forwardComputation(inputVector);
            for (int m = 0; m < myNet->myLayerCount+1; ++m) {
                for (int n = 0; n < 2; ++n) {
                    qDebug() << "layersOutputs[][]" << layersOutputs[m][n];
                }
            }
            backwardComputation(layersOutputs, rightAnswer);

            double *networkOutput = layersOutputs[myNet->myLayerCount];
            double error = computeError(rightAnswer, networkOutput);
            averageError += error;

            int lengthLO = myNet->myLayerCount+1;
            for (int k = 1; k < lengthLO; ++k) {
                delete [] layersOutputs[k];
            }
            delete [] layersOutputs;
        }
    }
    return averageError / myTrainingSetSize;
}

void NeuralNetwork::Trainer::backwardComputation(double **layersOutputs, double *rightAnswer) {
    double **gradients = computeGradients(layersOutputs, rightAnswer);
    updateWeights(gradients, layersOutputs);

    //Debug
    for (int i = 0; i < myNet->myLayerCount; ++i) {
        for (int j = 0; j < 2; ++j) {
            qDebug() << "gradients[][] = " << gradients[i][j];
        }
    }

    int length = myNet->myLayerCount;
    for (int i = 0; i < length; ++i) {
        delete [] gradients[i];
    }
    delete [] gradients;
}

double** NeuralNetwork::Trainer::computeGradients(double **layersOutputs, double *rightAnswer) const {
    double **gradients = new double * [myNet->myLayerCount];
    int lastIndex = myNet->myLayerCount - 1;
    double *prevGradient = computeGradientForOutputLayer(layersOutputs[myNet->myLayerCount], rightAnswer);
    gradients[lastIndex] = prevGradient;
    for (int i = lastIndex; i > 0; --i) {
        double *weightedSum = myNet->myLayer[i]->backPropagation(prevGradient);
        prevGradient = computeGradientForHiddenLayer(myNet->myLayer[i - 1], weightedSum, layersOutputs[i]);
        gradients[i - 1] = prevGradient;
        delete [] weightedSum;
    }
    return gradients;
}

double* NeuralNetwork::Trainer::computeGradientForOutputLayer(double *layerOutput, double *rightAnswer) const {
    int outputVectorDimension = myRightAnswersLength;
    double *g = new double[outputVectorDimension];
    for (int i = 0; i < outputVectorDimension; ++i) {
        double answer = layerOutput[i];
        double e = rightAnswer[i] - answer;
        g[i] = answer * (1 - answer) * e;
    }
    return g;
}

double* NeuralNetwork::Trainer::computeGradientForHiddenLayer(Layer *layer, double *weightedSum, double *layerOutput) const {
    int vectorDimension = layer->getSize();
    double *g = new double[vectorDimension];
    for (int i = 0; i < vectorDimension; ++i) {
        g[i] = layerOutput[i] * (1 - layerOutput[i]) * weightedSum[i+1];
    }
    return g;
}

void NeuralNetwork::Trainer::updateWeights(double **gradients, double **layersOutputs) {
    int length = myNet->myLayerCount;
    for (int i = 0; i < length; ++i) {
        Layer *layer = myNet->myLayer[i];
        int lengthLayer = layer->getPrevSize();
        int inputLength = layer->getSize();
        double **deltas = computeDeltas(gradients[i], lengthLayer, layersOutputs[i], inputLength, myPreviousDeltas[i]);

        // Debug
        for (int k = 0; k < lengthLayer; ++k) {
            for (int j = 0; j < inputLength; ++j) {
                qDebug() << "deltas[][] = " << deltas[k][j];
            }
        }

        layer->updateWeights(deltas, myRegularizationParameter);
        myPreviousDeltas[i] = deltas;
    }
}

double** NeuralNetwork::Trainer::computeDeltas(double *gradient, int length,
                                               double *layerInput, int inputLength, double **previousDelta) const {
    double **deltas = new double * [length];
    for (int i = 0; i < length; ++i) {
        deltas[i] = new double [inputLength];
    }
    for (int j = 0; j < inputLength; ++j) {
        double v = myRateOfLearning * gradient[j];

        deltas[0][j] = v;
        for (int i = 1; i < length; ++i) {
            deltas[i][j] = v * layerInput[i-1] + myMomentConstant * getPreviousDelta(previousDelta, i, j);
        }
    }
    return deltas;
}

double NeuralNetwork::Trainer::getPreviousDelta(double **previousDelta, int i, int j) const {
    return (previousDelta == NULL) ? 0 : previousDelta[i][j];
}

double NeuralNetwork::Trainer::computeError(double *rightAnswer, double *networkOutput) const {
    double error = 0;
    for (int k = 0; k < myRightAnswersLength; ++k) {
        double e = networkOutput[k] - rightAnswer[k];
        error += e * e;
    }
    return error / 2;
}

//void NeuralNetwork::Trainer::shuffleTrainingSet() {
//    for (std::vector<double*> vector : myTrainingSet) {
//        Collections.shuffle(vector);
//    }
//}







/**

  Берём set (состоящий из массива символов)
  Подаём его на вход Тренера
  Считаем выход НС, нам надо будет сохранить все выходы и совокупные выходы.
  Попадает ли разность между целевым образцом и реальным выходом сети в допустимые рамки?
  Вычисляем ошибку для каждого выходго слоя
  Для каждого скрытого слоя вычисляем ошибки


  */
