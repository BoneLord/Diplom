#include "trainingsetelement.h"

TrainingSetElement::TrainingSetElement(double *element, int rightAnswer) {
    myElement = element;
    myRightAnswer = rightAnswer;
}

double* TrainingSetElement::getData() {
    return myElement;
}

int TrainingSetElement::getRightAnswer() {
    return myRightAnswer;
}
