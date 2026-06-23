#ifndef __TENSOR_H
#define __TENSOR_H

#include "stdio.h"
#include <stdlib.h>
#include <time.h>

typedef float tensor_data_type;

typedef enum create_tensor_mode
{
	random = 1,
	zero,
	one,
	eye
}create_tensor_mode;

typedef enum isgrad
{
	nograd = 0,
	grad = 1
}isgrad;

typedef struct tensor
{
	unsigned int w;
	unsigned int h;
	unsigned int dim;
	tensor_data_type* Data;
	unsigned char req_grad;
	tensor_data_type* grad;
}tensor,*tensor_handle;


tensor_handle create_tensor(unsigned int dim,unsigned int h,unsigned int w,create_tensor_mode mode,isgrad gradmode);//创建一个tensor
tensor_handle create_tensor_from_twodim_arr(unsigned int h,unsigned int w,isgrad gradmode,void* arr);//从二维数组创建一个tensor
tensor_handle remove_tensor(tensor_handle tensortoremove);//删除一个tensor
tensor_data_type get_tensor_data(tensor_handle tensortoshow,unsigned int dim,unsigned int h,unsigned int w);//获取tensor中某个元素的值
tensor_handle tensor_T(tensor_handle tensortoT);//将tensor转置
tensor_handle tensor_matrix_mul(tensor_handle tensorA,tensor_handle tensorB);//矩阵乘法A.B
tensor_handle tensor_per_val_mul(tensor_handle tensorA,tensor_handle tensorB);//同型矩阵每个元素对应相乘
tensor_handle tensor_per_val_sub(tensor_handle tensorA,tensor_handle tensorB);//同型矩阵每个元素对应相减
tensor_handle tensor_per_val_sum(tensor_handle tensorA,tensor_handle tensorB);//同型矩阵每个元素对应相加
tensor_handle tensor_sum_all(tensor_handle tensorA);//矩阵中每个元素相加，最终得到一个1x1矩阵
tensor_handle tensor_mul_k(tensor_handle tensorA,tensor_data_type k);//矩阵中每个元素乘上k
tensor_handle tensor_to_line(tensor_handle tensortoline);//将矩阵按行优先展开成一条直线
tensor_handle tensor_padding(tensor_handle tensorpad,unsigned int size,create_tensor_mode mode);//将矩阵填充成目标大小
void tensor_zero_grad(tensor_handle tensortozerograd);//清零矩阵梯度
void change_grad_mode(tensor_handle tensortochangegrad,isgrad gradmode);//切换求梯度状态
void tensor_to_grad(tensor_handle tensorupdatagrad,tensor_handle gradtensor);//将tensor复制进另一个tensor的梯度
void tensor_sum_to_grad(tensor_handle tensorupdatagrad,tensor_handle gradtensor);//将tensor加进另一个tensor的梯度
tensor_handle grad_to_tensor(tensor_handle tensorA);//将梯度变成tensor
void show_tensor(tensor_handle tensortoshow);//打印tensor
void show_tensor_grad(tensor_handle tensortoshow);//打印梯度

#endif
