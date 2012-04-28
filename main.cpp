#include <QtCore/QCoreApplication>
#include "neuralnetwork.h"
#include <QDebug>
#include "trainingsetelement.h"
#include <fstream>
#include <iostream>
#include <string>

int** readFile(const char *outputFileName);
TrainingSetElement** initialization();
char* toNumber(int number);

int main(void) {

    int countLayers = 3;
    int countSymbol = 41;
    int inputSize = 14 * 14;
    int lengthSet = 13;
    int setSize = countSymbol * lengthSet;
    int * neuroNetSize;
    neuroNetSize = new int [countLayers];
    neuroNetSize[0] = inputSize;
    neuroNetSize[1] = countSymbol;
    neuroNetSize[2] = countSymbol;

    NeuralNetwork myNet(countLayers, neuroNetSize, 0, 1);

    TrainingSetElement **trainSet = initialization();

//    for (int i = 0; i < setSize; ++i) {
//        for (int j = 0; j < 14*14; ++j) {
//            std::cout << trainSet[i]->getData()[j] << " ";
//            if ((j+1) % 14 == 0) {
//                std::cout << std::endl;
//            }
//        }
//        std::cout << std::endl;
//    }
    char fileName[] = "weights.dat";
//    myNet.setWeightsFromFile(fileName);

    myNet.train(trainSet,setSize,0.25,0);

//    myNet.writeWeightsToFile(fileName);

    std::cout << "Active neuron = " << myNet.recognize(trainSet[0]->getData()) << std::endl;

    delete [] neuroNetSize;
    for (int i = 0; i < setSize; ++i) {
        delete trainSet[i];
    }
    delete [] trainSet;
    return 0;
}

TrainingSetElement** initialization() {
    int lengthSet = 13;
    int countSymbol = 41;
    int symbolSize = 14*14;
    int setSize = countSymbol * lengthSet;
    TrainingSetElement **trainSet = new TrainingSetElement * [setSize];

    int startIndex = 0;
    for (int k = 0; k < lengthSet; ++k) {
        std::string fileName = "trainingsets/trainingSet";
        char *number = toNumber(k);
        fileName.append(number);
        fileName.append(".txt");
        delete [] number;

        int **trainingSet = readFile(fileName.c_str());

        int stopIndex = startIndex + countSymbol;
        int i = 0;
        while (startIndex < stopIndex) {
            int *input = trainingSet[i];
            trainSet[startIndex] = new TrainingSetElement(symbolSize,input,i);
            ++startIndex;
            ++i;
        }
        startIndex = stopIndex;

        for (int j = 0; j < countSymbol; ++j) {
            delete [] trainingSet[j];
        }
        delete [] trainingSet;
    }
    return trainSet;
}

int** readFile(const char *outputFileName) {
    std::ifstream inputFile(outputFileName, std::ios::in | std::ios::binary);
    if (!inputFile.is_open()) {
        return NULL;
    }
    int length = 41;
    int symbolSize = 14*14;
    int **trainingSet = new int * [length];
    for (int i = 0; i < length; ++i) {
        trainingSet[i] = new int [symbolSize];
        inputFile.read((char *) trainingSet[i], symbolSize * sizeof(int));
    }
    inputFile.close();
    return trainingSet;
}

char* toNumber(int number) {
    std::string str = "";
    while (number / 10 != 0) {
        char symbol = (number % 10) + '0';
        str.insert(0, 1, symbol);
        number /= 10;
    }
    str.insert(0, 1, number + '0');    
    char * numStr;
    numStr = new char [str.size()+1];
    strcpy (numStr, str.c_str());
    return numStr;
}
