#ifndef _COMPUTE_H
#define _COMPUTE_H
#include "tensor.h"
#include <math.h>

typedef enum actifun_type
{
	relu = 1,
	sigmoid,
	none
}actifun_type;

typedef tensor_handle (*active_func)(tensor_handle);
typedef tensor_handle (*active_func_d)(tensor_handle,tensor_handle);
typedef float (*lose_func)(tensor_handle,tensor_handle);
typedef tensor_handle (*lose_func_d)(tensor_handle,tensor_handle);

tensor_handle tensor_sigmoid(tensor_handle t);
tensor_handle tensor_sigmoid_grad(tensor_handle grad, tensor_handle output);
tensor_handle tensor_relu(tensor_handle t);
tensor_handle tensor_relu_grad(tensor_handle grad, tensor_handle output);

float MSE(tensor_handle outputval,tensor_handle tureval);
tensor_handle MSE_d(tensor_handle outputval, tensor_handle trueval);

#endif
