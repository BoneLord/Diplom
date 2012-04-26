#include "trainingsetelement.h"

TrainingSetElement::TrainingSetElement(int size, double *element, int rightAnswer) {
    mySize = size;
    myRightAnswer = rightAnswer;
    myElement = new double [mySize];
    for (int i = 0; i < mySize; ++i) {
        myElement[i] = element[i];
    }
}

TrainingSetElement::TrainingSetElement(int size, int *element, int rightAnswer) {
    mySize = size;
    myRightAnswer = rightAnswer;
    myElement = new double [mySize];
    for (int i = 0; i < mySize; ++i) {
        myElement[i] = double(element[i]);
    }
}

TrainingSetElement::~TrainingSetElement() {
    delete [] myElement;
}

double* TrainingSetElement::getData() {
    return myElement;
}

int TrainingSetElement::getRightAnswer() const {
    return myRightAnswer;
}

int TrainingSetElement::getSize() const {
    return mySize;
}
