#include <stdio.h>
#include "tensor.h"

int main() {
    printf("🧠 Launching LinearTensor-Edge Neural Layer Micro-Kernel...\n\n");

    // Simulated 1x3 sensor payload array [Temp, Vibration, Current]
    FlatTensor inputs = initialize_tensor_space(1, 3);
    inputs.memory_vector[0] = 0.85; 
    inputs.memory_vector[1] = -0.40; 
    inputs.memory_vector[2] = 0.15;

    // Simulated 3x1 configuration weight matrix down-sampled from an external training set
    FlatTensor weights = initialize_tensor_space(3, 1);
    weights.memory_vector[0] = 0.75; 
    weights.memory_vector[1] = 1.10; 
    weights.memory_vector[2] = -0.35;

    printf("📥 Sensor Packet Linear Vector Mapping: [ 0.85, -0.40, 0.15 ]\n");

    FlatTensor forward_pass_node = calculate_layer_inference(&inputs, &weights);
    printf("💥 Pre-Activation Calculated Product Dot Score: %.4f\n", forward_pass_node.memory_vector[0]);

    apply_activation_sigmoid(&forward_pass_node);
    printf("🔮 Terminal Post-Sigmoid Predictive Status Probability: %.4f\n", forward_pass_node.memory_vector[0]);

    destroy_tensor_space(&inputs);
    destroy_tensor_space(&weights);
    destroy_tensor_space(&forward_pass_node);
    return 0;
}
