#include <QtCore/QCoreApplication>
#include "neuralnetwork.h"
#include <QDebug>
#include "trainingsetelement.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    int countLayers = 3;
    int * neuroNetSize;
    neuroNetSize = new int [countLayers];
    neuroNetSize[0] = 2;
    neuroNetSize[1] = 2;
    neuroNetSize[2] = 1;

    NeuralNetwork myNet(countLayers, neuroNetSize, 0, 1);

    int lengthInput = 2;
    double * input;
    input = new double [lengthInput];
    input[0] = 0.1;
    input[1] = 0.9;

    int setSize = 1;
    TrainingSetElement **trainSet = new TrainingSetElement * [setSize];
    for (int i = 0; i < setSize; ++i) {
        trainSet[i] = new TrainingSetElement(input,0);
    }

    myNet.setTestWeights();

    myNet.train(trainSet,setSize,0,0);

    qDebug() << "Active neuron = "<< myNet.recognize(input);
    //myNet.TestBackPropagation();

    delete [] neuroNetSize;
    delete [] input;

    return 0;
//    return a.exec();
}
