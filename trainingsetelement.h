#ifndef TRAININGSETELEMENT_H
#define TRAININGSETELEMENT_H

class TrainingSetElement
{
private:
    double *myElement;
    int myRightAnswer;
    TrainingSetElement(const TrainingSetElement& another);
    TrainingSetElement& operator=(const TrainingSetElement &another);
public:
    TrainingSetElement(double *element, int rightAnswer);
    double* getData();
    int getRightAnswer();
};

#endif // TRAININGSETELEMENT_H
