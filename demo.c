/*
License: This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.


MANUAL:
- INPUT: Inputs which are needed to be determinded by the user

- Compile: `make`

- Execute: `./sample`
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "scratchANN.h"
#include <omp.h>    // for parallel computation
#include <math.h>





int main()
{

    TrainSetting setting;

    // --------------------------------------------------------------------------------
    // --------------------------------- INPUTs ---------------------------------------
    // --------------------------------------------------------------------------------

    /* INPUT: the number of layers and neurons in each layer are given by `layers[]`
    Energy Efficiency Dataset had 8 inputs and 2 outputs,
    so the number of neurons in the first layer = 8, the number of neurons in the first layer = 2
    In the following example, we have 3 hidden layer in between, with 32, 16, and 8 neurons*/
    int layers[] = {8, 32, 16, 8, 2};
 
    /* INPUT:
    for example, = 0.8 means 80% of data for train and 20% for validation (actaully for test), I just called it here validation*/
    double train_split = 0.8;

    // INPUT: Define the batch size (int type), a higher batch size. memory usage will be higher
    setting.batch_size = 32;

    // INPUT (double type), learning rate in SGD
    setting.learning_rate = 0.001; 

    // INPUT (int type)
    setting.epochs = 20;

    /* INPUT: print metrics every PRINT_EVERY epochs (int type). 
    It will also print the first and last epoch results as well*/
    setting.PRINT_EVERY = 1;


    /* INPUT: type of outputs defines the metrics needed to be printed
    Options: "continuous" or "binary" or "categorical"
    Energy Efficiency Dataset has 2 outputs and each one are floating-point numbers (continuous)*/
    strcpy(setting.outputType, "continuous");


    /* INPUT: the name of file holding data.
    Energy Efficiency Dataset is saved into energy_efficiency_data.txt already available in the currect directory.
    The .txt file holds the data in a format that function `ReadDataFromFile(const char *filename, int *MAX_ROWS, int MAX_COLS)` is able to read
    (In file there is a TAB sapce (\t) between each column and lines starting with '#' will not be read). check out the file to see the data.*/
    char filename[] = "energy_efficiency_data.txt";

    /* INPUT: Set to true to shuffle the data
    `ReadDataFromFile()` will return a 2D array called data. Before splitting the data into train and validation (or test) it is better to be shuffled*/
    bool shuffle = true;

    /* INPUT: Set true to standardize inputs (X_train and X_val) to the model
    A function called `standardize_features()` standardize the data based on standard deviation to above instability of data especially in Energy Efficiency Dataset.*/
    bool scale_standardize = true; 

    // 
    /* INPUT: the name of a text file to save the trained model on disk.
    Users can load the model to use or re-train it with same or different data and settings.*/
    char filnameSave[] = "model.txt";
    
    // INPUT: Number of threads in parallel computation
    omp_set_num_threads(6);


    //**********************************************************************
    //********************* activation function setup **********************
    //**********************************************************************
    // IMPORTANT Note: Errors in inputs are not handled. Please make sure inputs for activation functions are correct

    /* INPUT:
    In layer2 (second layer or first hidden layer),
    0.25/(0.25+0.3+0.5)*100 precentage of neurons have `linear` activation function,
    0.3/(0.25+0.3+0.5)*100 precentage of neurons have `relu` activation function,
    0.5/(0.25+0.3+0.5)*100 precentage of neurons have `tanh_function` activation function:*/
    ActivationConfig layer2_configs[] = {
        {linear, 0.25},{relu, 0.3}, {tanh_function, 0.5}}; 

    /* INPUT:
    In layer3 (third layer or second hidden layer),
    100 precentage of neurons have `tanh_function` activation function: */
    ActivationConfig layer3_configs[] = {
        {tanh_function, 1}};

    /* INPUT:
    In layer4 (fourth layer or third hidden layer),
    5/(5+3)*100 precentage of neurons have `sigmoid` activation function,
    3/(5+3)*100 precentage of neurons have `tanh_function` activation function: */
    ActivationConfig layer4_configs[] = {
        {sigmoid, 5},{tanh_function, 3}};
    
    /* INPUT:
    The output layer (layer5),
    1/(1+1) proportation of neurons assigned to `linear` activation function,
    1/(1+1) proportation of neurons assigned to have `relu` activation function:
    */
    ActivationConfig layer5_configs[] = {
        {relu, 1},{linear, 1}};
    
    /*Since the last layer has 2 neurons (refer to `layers[]`),
    one of the neurons will have `linear` and the other one will have `relu` activation function.
    We could just use `linear` for both, but here I wanted to show you how to give the inputs.
    We can use `relu` as both outputs have only positive values.
    */
    
    /* INPUT:
    Manually input the size of each config array in `config_sizes[]`.
    In `config_sizes[]`:
    the number `3` shows the number of two configs for first hidden layer (layer2)
    the number `2` shows the number of two configs for last layer (`layer5_configs`)*/
    int config_sizes[] = {3, 1, 2, 2}; 
    
    /*INPUT:
    Essentially, insert the inputs (`layer{i}_configs`) manually into `*activation_configs[]` */
    ActivationConfig *activation_configs[] = {layer2_configs, layer3_configs, layer4_configs, layer5_configs};
    
    
    // --------------------------------------------------------------------------------
    // ------------------------------ END OF INPUTs -----------------------------------
    // --------------------------------------------------------------------------------





    // ------------------------- Computed based on Inputs -----------------------------
    int nLayers = sizeof(layers) / sizeof(layers[0]);
    int num_inputs = layers[0];
    int num_outputs = layers[nLayers - 1];
    int NUM_FEATURES = num_inputs + num_outputs; // or number of columns in data.txt
    int NUM_SAMPLES = 0;                         // or the number of rows will be read from data.txt
    // --------------------------------------------------------------------------------

    // ------------------------------- Read Data --------------------------------------
    printf("Reading data...\n");
    double **data = ReadDataFromFile(filename, &NUM_SAMPLES, NUM_FEATURES);
    
    // --------------------------------------------------------------------------------

    // ------------- Computed based on Inputs and ReadDataFromFile()-------------------
    int num_train = (int)(NUM_SAMPLES * train_split);
    int num_val = NUM_SAMPLES - num_train;
    // --------------------------------------------------------------------------------

    // -------------------------- Shuffle and Split Data ------------------------------
    // Shuffle the data if requested
    if (shuffle)
    {
        shuffleData(data, num_train + num_val, num_inputs + num_outputs);
    }

    double **X_train = NULL, **Y_train = NULL;
    double **X_val = NULL, **Y_val = NULL;

    printf("Splitting data into train and validation dataset...\n");
    splitData(data, num_inputs, num_outputs, num_train, num_val, &X_train, &Y_train, &X_val, &Y_val);

    // USER: based on splitData I won't need the original data
    deallocate2DArray(data, NUM_SAMPLES);

    if (scale_standardize)
    {
        standardize_features(X_train, X_val, num_train, num_val, num_inputs);
    }
    
    // --------------------------------------------------------------------------------

    // ----------------------------- Creat the model ----------------------------------
    NeuralNetwork *nn = createModel(layers, nLayers, activation_configs, config_sizes);
    summary(nn);
    // --------------------------------------------------------------------------------

    // ----------------------------- Train the model ----------------------------------
    // double start, end, cpu_time_used;
    // start = omp_get_wtime();

    trainModel(nn, X_train, Y_train, num_train, X_val, Y_val, num_val, setting);

    // end = omp_get_wtime();
    // cpu_time_used = end - start;
    // printf("Wall Clock Time: %f seconds\n", cpu_time_used);
    // --------------------------------------------------------------------------------

    // ----------------------------------- Save ---------------------------------------
    printf("\nSaving the model on disk...\n");
    saveModel(nn, filnameSave);
    // --------------------------------------------------------------------------------

    // ------------------------ Free memory for the model -----------------------------
    printf("De-allocating the memory for ANN model...\n");
    freeModel(nn);
    // --------------------------------------------------------------------------------

    // ----------------------------------- Load ---------------------------------------
    printf("Loading the model from disk...\n");
    NeuralNetwork *loaded_nn = loadModel(filnameSave);
    // --------------------------------------------------------------------------------

    // --------------------- Test laoded model by re-training -------------------------
    printf("IMPORTANT: Testing loaded model by training for new settings...\n");

    setting.learning_rate = 0.0001;
    setting.epochs = 2;
    trainModel(loaded_nn, X_train, Y_train, num_train, X_val, Y_val, num_val, setting);
    
    // --------------------------------------------------------------------------------

    // ------------------------ Free memory for the model -----------------------------
    printf("De-allocating the memory for ANN model...\n");
    freeModel(loaded_nn);
    // --------------------------------------------------------------------------------

    // -------------------------------- Free Data -------------------------------------
    printf("De-allocating train and validation data.\n");
    deallocate2DArray(X_train, num_train);
    deallocate2DArray(Y_train, num_train);
    deallocate2DArray(X_val, num_val);
    deallocate2DArray(Y_val, num_val);
    // --------------------------------------------------------------------------------

    return 0;
}

