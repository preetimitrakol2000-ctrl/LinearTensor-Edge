#include <stdlib.h>
#include <math.h>
#include "tensor.h"

FlatTensor initialize_tensor_space(int rows, int cols) {
    FlatTensor t;
    t.row_count = rows; t.col_count = cols;
    t.memory_vector = (double*)malloc(rows * cols * sizeof(double));
    return t;
}

FlatTensor calculate_layer_inference(FlatTensor* inputs, FlatTensor* weights) {
    FlatTensor result_matrix = initialize_tensor_space(inputs->row_count, weights->col_count);
    
    for (int i = 0; i < inputs->row_count; i++) {
        for (int j = 0; j < weights->col_count; j++) {
            double scalar_accumulator = 0.0;
            for (int k = 0; k < inputs->col_count; k++) {
                // Flattened 1D matrix multiplication formula
                scalar_accumulator += inputs->memory_vector[i * inputs->col_count + k] * weights->memory_vector[k * weights->col_count + j];
            }
            result_matrix.memory_vector[i * result_matrix.col_count + j] = scalar_accumulator;
        }
    }
    return result_matrix;
}

void apply_activation_sigmoid(FlatTensor* t) {
    for (int i = 0; i < (t->row_count * t->col_count); i++) {
        t->memory_vector[i] = 1.0 / (1.0 + exp(-t->memory_vector[i]));
    }
}

void destroy_tensor_space(FlatTensor* t) {
    free(t->memory_vector);
}
