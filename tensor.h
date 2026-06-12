#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int row_count;
    int col_count;
    double* memory_vector;
} FlatTensor;

FlatTensor initialize_tensor_space(int rows, int cols);
void destroy_tensor_space(FlatTensor* t);
FlatTensor calculate_layer_inference(FlatTensor* inputs, FlatTensor* weights);
void apply_activation_sigmoid(FlatTensor* t);

#endif
