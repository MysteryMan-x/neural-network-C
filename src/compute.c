#include "compute.h"

tensor_handle tensor_sigmoid(tensor_handle t)
{
	tensor_handle out = create_tensor(t->dim, t->h, t->w, zero, t->req_grad);
	for (int i = 0; i < t->h * t->w; i++) {
		float x = t->Data[i];
		out->Data[i] = 1.0f / (1.0f + exp(-x));
	}
	return out;
}

tensor_handle tensor_sigmoid_grad(tensor_handle grad, tensor_handle output)
{
	tensor_handle out = create_tensor(grad->dim, grad->h, grad->w, zero, grad->req_grad);
	for (int i = 0; i < grad->h * grad->w; i++) {
		float s = output->Data[i];
		out->Data[i] = grad->Data[i] * s * (1.0f - s); 
	}
	return out;
}

tensor_handle tensor_relu(tensor_handle t)
{
	tensor_handle out = create_tensor(t->dim, t->h, t->w, zero, t->req_grad);
	for (int i = 0; i < t->h * t->w; i++) {
		out->Data[i] = t->Data[i] > 0 ? t->Data[i] : 0;
	}
	return out;
}

tensor_handle tensor_relu_grad(tensor_handle grad, tensor_handle output)
{
	tensor_handle out = create_tensor(grad->dim, grad->h, grad->w, zero, grad->req_grad);
	for (int i = 0; i < grad->h * grad->w; i++) {
		out->Data[i] = output->Data[i] > 0 ? grad->Data[i] : 0;
	}
	return out;
}

// MSE 均方误差损失函数：0.5 * sum(output - true)²
float MSE(tensor_handle outputval, tensor_handle trueval)
{
	float loss = 0.0f;
	tensor *sub, *mul_square, *mul_05, *sum;
	sub = tensor_per_val_sub(outputval, trueval);
	mul_square = tensor_per_val_mul(sub, sub);
	sum = tensor_sum_all(mul_square);
	mul_05 = tensor_mul_k(sum, 0.5f);
	
	loss = mul_05->Data[0];
	
	remove_tensor(sub);
	remove_tensor(mul_square);
	remove_tensor(sum);
	remove_tensor(mul_05);
	
	return loss;
}

// MSE 反向传播导数：d(Loss)/d(output) = output - true
tensor_handle MSE_d(tensor_handle outputval, tensor_handle trueval)
{
	return tensor_per_val_sub(outputval, trueval);
}

