#ifndef TRAININGSETELEMENT_H
#define TRAININGSETELEMENT_H

class TrainingSetElement
{
private:
    double *myElement;
    int myRightAnswer;
    int mySize;
    TrainingSetElement(const TrainingSetElement& another);
    TrainingSetElement& operator=(const TrainingSetElement &another);
public:
    TrainingSetElement(int size, double *element, int rightAnswer);
    TrainingSetElement(int size, int *element, int rightAnswer);
    ~TrainingSetElement();
    double* getData();
    int getRightAnswer() const;
    int getSize() const;
};

#endif // TRAININGSETELEMENT_H
