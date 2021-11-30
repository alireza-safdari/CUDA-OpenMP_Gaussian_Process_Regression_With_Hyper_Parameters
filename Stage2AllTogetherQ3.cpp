// Stage1KOnMyPC.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <iostream>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <sys/time.h>
#include <omp.h>
#include <string>
#include <math.h>


struct MatrixDataForMultiplication
{
    double* matrix;
    uint64_t iSize, jSize;
};



double get5HundredthsRandom(void)
{
    return  (0.100 * (double)rand() / (double)RAND_MAX) - 0.05 ;
}


uint32_t getARandomNumber(uint32_t min_, uint32_t max_) // max not included
{
    uint32_t range = max_ - min_;
    uint32_t r1 = rand();
    uint32_t r2 = rand();
    return ((r1 * r2) % range) + min_; // to randoms are used to extend the range of the random number. the rand has a max of 32767
}



const uint32_t COLUM_NOT_AVAILABLE = 0xFFFFFFFF;
bool columnAvailabe(int n) {
    return (n != COLUM_NOT_AVAILABLE);
}


void printMatrix(const double matrix[], const uint32_t iSize, const uint32_t jSize)
{
    uint32_t index = 0;
    for (uint32_t i = 0; i < iSize; i++)
    {
        for (uint32_t j = 0; j < jSize; j++)
        {
            printf("  %lf", matrix[index]);
            index++;
        }
        printf("\n");
    }
}


void printMatrix(const uint32_t matrix[], const uint32_t iSize, const uint32_t jSize)
{
    uint32_t index = 0;
    for (uint32_t i = 0; i < iSize; i++)
    {
        for (uint32_t j = 0; j < jSize; j++)
        {
            printf("  %d", matrix[index]);
            index++;
        }
        printf("\n");
    }
}


void multiplyMatrixParallel(const MatrixDataForMultiplication& matrixA,
                           const MatrixDataForMultiplication& matrixB,
                           MatrixDataForMultiplication& result)
{
    result.iSize = matrixA.iSize;
    result.jSize = matrixB.jSize;
    #pragma omp parallel shared(matrixA, matrixB, result)
    {
        #pragma omp for nowait collapse(2)
        for (uint64_t i = 0; i < result.iSize; i++)
        {
            for (uint64_t j = 0; j < result.jSize; j++)
            {
                result.matrix[i * result.jSize + j] = 0;
                for (uint64_t k = 0; k < matrixA.jSize; k++)
                {
                    // printf("%.2lf X %.2lf + ", matrixA[matrixAIndex], matrixB[matrixBIndex]);
                    result.matrix[i * result.jSize + j] += matrixA.matrix[i * matrixA.jSize + k] * matrixB.matrix[k * matrixB.jSize + j];
                }
            }
        }
    }
}




void computeLUMatrixCompact(double matrix[], const uint64_t n)
{
    for (uint64_t k = 0; k < n - 1; k++)
    {
        uint64_t chunkSize = 100000;
        if( (chunkSize / (n - k)) < 20)
        {
            chunkSize = chunkSize / (n - k);
            #pragma omp parallel shared(matrix, k, n, chunkSize)
            {
                #pragma omp for nowait schedule(static,chunkSize)
                    for (uint64_t i = k + 1; i < n; i++)
                    {
                        double toBeL = matrix[i * n + k] / matrix[k * n + k];
                        matrix[i * n + k] = toBeL;
                        for (uint64_t j = k + 1; j < n; j++)
                        {
                            matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
                        }
                    }
            }
        }
        else
        {
            #pragma omp parallel shared(matrix, k, n)
            {
                #pragma omp for nowait schedule(static)
                    for (uint64_t i = k + 1; i < n; i++)
                    {
                        double toBeL = matrix[i * n + k] / matrix[k * n + k];
                        matrix[i * n + k] = toBeL;
                        for (uint64_t j = k + 1; j < n; j++)
                        {
                            matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
                        }
                    }
            }
        }
    }
}



// AX = B
// A = LU
// LUX = B
// We can solve that by solving the 2 below
// 1) LY = B
// 2) UX = Y
// In this function solving for 1)
// size of compactMatrix is nXn while B and Y are nX1,  Y is unkown
void substitutionForL(double compactMatrix_[], double matrixB_[], double matrixY_[], const uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        double rightSide = matrixB_[i]; // we make it so that "Y[i] = rightSide" mathematically
        for (uint64_t j = 0; j < i; j++)
        {
            rightSide -= compactMatrix_[i * n + j] * matrixY_[j];
        }
        matrixY_[i] = rightSide;
    }
}


// AX = B
// A = LU
// LUX = B
// We can solve that by solving the 2 below
// 1) LY = B
// 2) UX = Y
// In this function solving for 2)
// size of compactMatrix is nXn while y and X are nX1, X is unkown
void substitutionForU(double compactMatrix_[], double matrixY_[], double matrixX_[], const uint64_t n)
{
    for (int64_t i = n - 1; i > -1; i--)
    {
        double rightSide = matrixY_[i]; // we make it so that "Y[i] = rightSide" mathematically
        for (uint64_t j = n - 1; j > i; j--)
        {
            rightSide -= compactMatrix_[i * n + j] * matrixX_[j];
        }
        matrixX_[i] = rightSide / compactMatrix_[i * n + i];
    }
}



void fFunction(double* result, double* pointX, double* pointY, uint32_t nPoints, bool noise)
{
    if (noise)
    {
        #pragma omp parallel for 
        for (uint32_t i = 0; i < nPoints; i++)
        {
            double xPart = pointX[i] - 0.5;
            xPart = xPart * xPart * xPart * xPart;
            double yPart = pointY[i] - 0.5;
            yPart = yPart * yPart * yPart * yPart;
            double xyDependentPart = xPart + yPart;

            result[i] = -20 * xyDependentPart;
            result[i] += get5HundredthsRandom(); // adding noise
        }
    }
    else
    {
        #pragma omp parallel for 
        for (uint32_t i = 0; i < nPoints; i++)
        {
           double xPart = pointX[i] - 0.5;
            xPart = xPart * xPart * xPart * xPart;
            double yPart = pointY[i] - 0.5;
            yPart = yPart * yPart * yPart * yPart;
            double xyDependentPart = xPart + yPart;

            result[i] = -20 * xyDependentPart;
        }
    }
}




int main(int argc, char* argv[])
{
    uint32_t m = 50;
    if (argc == 2)
    {
        m = std::stol(argv[1], nullptr, 10);
    }

    struct timeval start, stop;
    double total_time;

    gettimeofday(&start, NULL); 

    const double distanceIncrement = 1.000 / static_cast<double>(m + 1);

    const uint32_t n = m * m;
    const uint32_t nPoints = m * m;
    const uint32_t nTestPoints = (m * m) / 10;
    const uint32_t nTrainingPoints = nPoints - nTestPoints;

    double* distanceXY = new double[m];

    uint32_t* cumulativeTestPointRowByRow = new uint32_t[m];

    uint32_t* indicesInThisRow = new uint32_t[m];

    uint32_t* testIndices = new uint32_t[nTestPoints];
    double* testPointX = new double[nTestPoints];
    double* testPointY = new double[nTestPoints];

    uint32_t* trainIndices = new uint32_t[nTrainingPoints];
    double* trainPointX = new double[nTrainingPoints];
    double* trainPointY = new double[nTrainingPoints];

    double* observedValues = new double[nTrainingPoints];
    double* fValuesForTestPoints = new double[nTestPoints];
    
    double L1 = 0.1;
    double L2 = 0.1;
    double lIncrement = 0.1;
    uint32_t nLValues = 10;

    double* trainingK = new double[nTrainingPoints * nTrainingPoints];

    double* kStarTranspose = new double[nTestPoints * nTrainingPoints];

    // for substitution after LU factorization
    double* matrixY = new double[nTrainingPoints * nTrainingPoints]; 
    double* matrixX = new double[nTrainingPoints * nTrainingPoints]; 

    double* testPointEstimation = new double[nTestPoints];

    double* mseValues = new double[nLValues * nLValues];


    distanceXY[0] = distanceIncrement;
    for (uint32_t i = 1; i < m; i++)
    {
        distanceXY[i] = distanceXY[i - 1] + distanceIncrement;
    }

    cumulativeTestPointRowByRow[0] = nTestPoints / m;
    for (uint32_t i = 1; i < m; i++)
    {
        uint32_t nPointsForThisRow = (nTestPoints - cumulativeTestPointRowByRow[i - 1]) / (m - i);
        cumulativeTestPointRowByRow[i] = cumulativeTestPointRowByRow[i - 1] + nPointsForThisRow;
    }
    
    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < m; j++)
        {
            indicesInThisRow[j] = (i * m) + j;
        }

        uint32_t nTestPointInThisRow = cumulativeTestPointRowByRow[0];
        uint32_t testIndex = 0;
        if (i != 0)
        {
            nTestPointInThisRow = cumulativeTestPointRowByRow[i] - cumulativeTestPointRowByRow[i - 1];
            testIndex = cumulativeTestPointRowByRow[i - 1];
        }

        std::fill(&testPointY[testIndex], &testPointY[testIndex + nTestPointInThisRow], distanceIncrement * static_cast<double>(i + 1));
        for (uint32_t j = 0; j < nTestPointInThisRow; j++)
        {
            uint32_t randomColumnForTraining;
            do
            {
                randomColumnForTraining = getARandomNumber(0, m);
            } while (indicesInThisRow[randomColumnForTraining] == COLUM_NOT_AVAILABLE);

            testIndices[testIndex] = indicesInThisRow[randomColumnForTraining];
            testPointX[testIndex] = distanceXY[randomColumnForTraining]; // distanceIncrement * static_cast<double>(randomColumnForTraining + 1);
            testPointY[testIndex] = distanceXY[i]; // distanceIncrement* static_cast<double>(i + 1);
            indicesInThisRow[randomColumnForTraining] = COLUM_NOT_AVAILABLE;
            testIndex++;
        }

        uint32_t trainIndex = 0;
        if (i != 0)
        {
            trainIndex = (i * m) - cumulativeTestPointRowByRow[i - 1];
        }
        for (uint32_t j = 0; j < m; j++)
        {
            if (indicesInThisRow[j] != COLUM_NOT_AVAILABLE)
            {
                trainIndices[trainIndex] = indicesInThisRow[j];
                trainPointX[trainIndex] = distanceXY[j];
                trainPointY[trainIndex] = distanceXY[i];
                trainIndex++;
            }
        }
    }

    // printf("distanceXY:\n");
    // printMatrix(distanceXY, 1, m);
    // printf("testIndices:\n");
    // printMatrix(testIndices, 1, nTestPoints);
    // printf("testPointX:\n");
    // printMatrix(testPointX, 1, nTestPoints);
    // printf("testPointY:\n");
    // printMatrix(testPointY, 1, nTestPoints);


    fFunction(observedValues, trainPointX, trainPointY, nTrainingPoints, true);
    // printf("\nobservedValues:\n");
    // printMatrix(observedValues, 1, nTrainingPoints);


    fFunction(fValuesForTestPoints, testPointX, testPointY, nTestPoints, false);
    // printf("\fValuesForTestPoints:\n");
    // printMatrix(fValuesForTestPoints, 1, nTestPoints);
    


    for (uint32_t l1Iterator = 0; l1Iterator < nLValues; l1Iterator++, L1 += lIncrement)
    {
        for (uint32_t l2Iterator = 0; l2Iterator < nLValues; l2Iterator++, L2 += lIncrement)
        {
            const double L1SquareMulti2 = L1 * L1 * 2;
            const double L2SquareMulti2 = L2 * L2 * 2;
            const double myPi = 3.141592653589793;
            const double oneOverSqrt2Pi = 1.0 / (sqrt(2.0 * myPi));
            #pragma omp parallel for collapse(1)
            for (uint32_t i = 0; i < nTrainingPoints; i++)
            {
                for (uint32_t j = 0; j < nTrainingPoints; j++) // approach 2
                {
                    double xDistancePart = trainPointX[i] - trainPointX[j];
                    xDistancePart = xDistancePart * xDistancePart;
                    xDistancePart /= L1SquareMulti2;

                    double yDistancePart = trainPointY[i] - trainPointY[j];
                    yDistancePart = yDistancePart * yDistancePart;
                    yDistancePart /= L2SquareMulti2;
                
                    double totalDistance = yDistancePart + xDistancePart;


                    double kValue = exp(-totalDistance) * oneOverSqrt2Pi;
                    trainingK[i * nTrainingPoints + j] = kValue;
                }
            }
            // printf("\ntrainingK before noise:\n");
            // printMatrix(trainingK, nTrainingPoints, nTrainingPoints);



            for (uint64_t i = 0; i < nTrainingPoints; i++)
            {
                trainingK[i * nTrainingPoints + i] += 0.01;
            }
            // printf("\ntrainingK after noise:\n");
            // printMatrix(trainingK, nTrainingPoints, nTrainingPoints);


            #pragma omp parallel for
            for (uint32_t i = 0; i < nTestPoints; i++)
            {
                for (uint32_t j = 0; j < nTrainingPoints; j++)
                {
                    double xDistancePart = trainPointX[j] - testPointX[i];
                    xDistancePart = xDistancePart * xDistancePart;
                    xDistancePart /= L1SquareMulti2;

                    double yDistancePart = trainPointY[j] - testPointY[i];
                    yDistancePart = yDistancePart * yDistancePart;
                    yDistancePart /= L2SquareMulti2;

                    double totalDistance = yDistancePart + xDistancePart;


                    double kValue = exp(-totalDistance) * oneOverSqrt2Pi;
                    kStarTranspose[i * nTrainingPoints + j] = kValue;
                }
            }
            // printf("\ntrainingK after noise:\n");
            // printMatrix(kStarTranspose, nTestPoints, nTrainingPoints);
            

            computeLUMatrixCompact(trainingK, nTrainingPoints);
            substitutionForL(trainingK, observedValues, matrixY, nTrainingPoints);
            substitutionForU(trainingK, matrixY, matrixX, nTrainingPoints);


            struct MatrixDataForMultiplication kTransposeCalculatedMultiplication = {kStarTranspose, nTestPoints, nTrainingPoints};
            struct MatrixDataForMultiplication matrixXMultiplication = {matrixX, nTrainingPoints, 1};
            struct MatrixDataForMultiplication testPointEstimationtMultiplication = {testPointEstimation, nTestPoints, 1};

            multiplyMatrixParallel(kTransposeCalculatedMultiplication, matrixXMultiplication, testPointEstimationtMultiplication);


            const uint32_t mseIndex = l1Iterator * nLValues + l2Iterator;
            mseValues[mseIndex]  = 0;
            for (uint32_t i = 0; i < nTestPoints; i++)
            {
                double delta = fValuesForTestPoints[i] - testPointEstimation[i];
                mseValues[mseIndex] += delta * delta;
            }
            mseValues[mseIndex] /= nTestPoints;
            // printf("mseValues: %lf, L1: %lf, L2: %lf\n", mseValues[mseIndex], L1, L2);
        }
    }

    // printf("mseValues:\n");
    // printMatrix(mseValues, nLValues, nLValues);
    

    double minimumMse = mseValues[0];
    double minimumL1 = 0.1, minimumL2 = 0.1;
    for (uint32_t l1Iterator = 0; l1Iterator < nLValues; l1Iterator++, L1 += lIncrement)
    {
        for (uint32_t l2Iterator = 0; l2Iterator < nLValues; l2Iterator++, L2 += lIncrement)
        {
            const uint32_t mseIndex = l1Iterator * nLValues + l2Iterator;
            if (minimumMse > mseValues[mseIndex])
            {
                minimumMse = mseValues[mseIndex];
                minimumL1 = (l1Iterator + 1) * lIncrement;
                minimumL2 = (l2Iterator + 1) * lIncrement;
            }
            
        }
    }
    // printf("\n\n minimum MSE: %lf, minimumL1: %lf, minimumL2: %lf\n", minimumMse, minimumL1, minimumL2);

    gettimeofday(&stop, NULL);
    total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);
    printf("For m = %d, it took %lf. Lowest MSE: %lf at L1: %.2lf and L2: %.2lf and for %d test points.\n", m, total_time, minimumMse, minimumL1, minimumL2, nTestPoints);

    delete[] distanceXY;

    delete[] cumulativeTestPointRowByRow;

    delete[] indicesInThisRow;
    
    delete[] testIndices;
    delete[] testPointX;
    delete[] testPointY;

    delete[] trainIndices;
    delete[] trainPointX;
    delete[] trainPointY;

    delete[] observedValues;
    delete[] fValuesForTestPoints;

    delete[] trainingK;

    delete[] kStarTranspose;

    delete[] matrixY;
    delete[] matrixX;

    delete[] testPointEstimation;

    delete[] mseValues;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
