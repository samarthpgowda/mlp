#ifndef MLP_H
#define MLP_H

#define INPUT_SIZE 241
#define HIDDEN_SIZE 16
#define FRAC_BITS 10

typedef short data_t;
typedef long long acc_t;

void mlp(data_t input[INPUT_SIZE], int &output);

#endif