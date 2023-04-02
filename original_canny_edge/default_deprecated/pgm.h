#pragma warning(disable : 4996)
#include <iostream>
#include <fstream>
#include <cstring>

class pgmImage
{
private:
    unsigned int rows;
    unsigned int cols;
    char pgmType[2];
    unsigned int maxValue;
public:
    pgmImage();
    double** matrix;
    void readImage(const char* fileName);
    void writeImage(const char* fileName, int** outputMatrix);
    void ignoreComments(FILE* pgmFile);
    double** allocateMatrix();
    void deallocateMatrix();
    unsigned int getHeight();
    unsigned int getWidth();
    char getType();
    unsigned int getMaxValue();
};

pgmImage::pgmImage() {
    matrix = NULL;
    rows = 0;
    cols = 0;
    pgmType[0] = ' ';
    pgmType[1] = ' ';
    maxValue = 0;
}

void pgmImage::readImage(const char* fileName) {
    FILE* pgmFile;
    pgmFile = fopen(fileName, "rb");
    if (pgmFile)
        std::cout << "File read successfully" << std::endl;
    else {
        std::cout << "File: '" << fileName << "' does not exist" << std::endl;
        exit(EXIT_FAILURE);
    }
    ignoreComments(pgmFile);

    fscanf(pgmFile, "%s", &pgmType);
    std::cout << pgmType << std::endl;
    if (strcmp(pgmType,"P5") != 0)
        std::cout << "Incorrect File Type: Expected P5" << std::endl;
    ignoreComments(pgmFile);

    fscanf(pgmFile, "%d %d", &cols, &rows);
    std::cout << "Width: " << cols << ", Height: " << rows << std::endl;
    ignoreComments(pgmFile);

    fscanf(pgmFile, "%d", &maxValue);
    std::cout << "Max Gray Value: " << maxValue << std::endl;
    fgetc(pgmFile);

    this->matrix = allocateMatrix();
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            this->matrix[i][j] = fgetc(pgmFile);
        }
    }

    fclose(pgmFile);
    return;
}

void pgmImage::writeImage(const char* fileName, int** outputMatrix) {
    FILE* pgmFile;

    pgmFile = fopen(fileName, "wb");
    if (pgmFile == NULL) {
        std::cout << "File: '" << fileName << "' failed to open" << std::endl;
        exit(EXIT_FAILURE);
    }

    fprintf(pgmFile, "P2\n");
    fprintf(pgmFile, "%d %d\n", this->cols, this->rows);
    fprintf(pgmFile, "255\n");
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            fprintf(pgmFile, "%d ", outputMatrix[i][j]);
        }
        fprintf(pgmFile, "\n");
    }

    fclose(pgmFile);
    return;
}

void pgmImage::ignoreComments(FILE* pgmFile) {
    int ch;
    char line[50];
    while ((ch = fgetc(pgmFile)) != EOF && isspace(ch));
    if (ch == '#') {
        fgets(line, sizeof(line), pgmFile);
        ignoreComments(pgmFile);
    }
    else
        fseek(pgmFile, -1, SEEK_CUR);

    return;
}

double** pgmImage::allocateMatrix() {
    double** newMatrix;

    newMatrix = (double**)malloc(sizeof(double*) * this->rows);
    if (newMatrix == NULL) {
        std::cout << "Error allocating memory" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < this->rows; i++) {
        newMatrix[i] = (double*)malloc(sizeof(double) * this->cols);
        if (newMatrix[i] == NULL) {
            std::cout << "Error allocating memory" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return newMatrix;
}

void pgmImage::deallocateMatrix() {
    for (int i = 0; i < this->rows; i++) {
        free(this->matrix[i]);
    }
    free(this->matrix);

    return;
}

unsigned int pgmImage::getHeight() {
    return this->rows;
}

unsigned int pgmImage::getWidth() {
    return this->cols;
}

char pgmImage::getType() {
    return this->pgmType[1];
}

unsigned int pgmImage::getMaxValue() {
    return this->maxValue;
}
