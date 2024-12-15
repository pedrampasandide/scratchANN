
#define _POSIX_C_SOURCE 200809L // getline() function in some OS creates a warning while compile so I add this
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <sys/types.h>

// Here write a function to read data.txt and save the values in data[MAX_ROWS][MAX_COLS];
double **ReadDataFromFile(const char *filename, int *MAX_ROWS, int MAX_COLS)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
        perror("Unable to open file");
        exit(EXIT_FAILURE);
    }

    double **data = NULL; // Pointer to hold the 2D array
    *MAX_ROWS = 0;        // Initialize the number of rows

    char *line = NULL;    // Pointer to the dynamic line buffer
    size_t line_size = 0; // Size of the allocated buffer
    ssize_t read;         // Number of characters read

    while ((read = getline(&line, &line_size, file)) != -1)
    {
        // Skip lines that start with '#'
        if (line[0] == '#')
        {
            continue;
        }

        // Allocate/reallocate memory for a new row
        data = realloc(data, (*MAX_ROWS + 1) * sizeof(double *));
        if (!data)
        {
            perror("Memory allocation failed");
            fclose(file);
            free(line);
            exit(EXIT_FAILURE);
        }

        // Allocate memory for the columns in the current row
        data[*MAX_ROWS] = malloc(MAX_COLS * sizeof(double));
        if (!data[*MAX_ROWS])
        {
            perror("Memory allocation failed");
            fclose(file);
            free(line);
            exit(EXIT_FAILURE);
        }

        // Parse the line into doubles and store in the current row
        char *token = strtok(line, "\t\n");
        int col = 0;
        while (token && col < MAX_COLS)
        {
            data[*MAX_ROWS][col] = atof(token);
            token = strtok(NULL, "\t\n");
            col++;
        }

        (*MAX_ROWS)++; // Increment the row count
    }

    free(line); // Free the line buffer
    fclose(file);

    printf("Data file read successfully.\n");
    printf("Number of samples:  %d\n", *MAX_ROWS);
    printf("Number of features: %d\n", MAX_COLS);
    printf("\n");
    return data;
}