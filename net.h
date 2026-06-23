#ifndef _NET_H
#define _NET_H

#include "tensor.h"
#include "compute.h"

typedef enum layer_type
{
	fc = 0,
	conv2d,
	pool
}layer_type;

typedef struct FC
{
	layer_type type;
	
	unsigned int input_node_num;
	unsigned int output_node_num;
	
	tensor* layer_w_tensor;
	tensor* layer_b_tensor;
}FC;

typedef struct CONV2D
{
	layer_type type;
	
	unsigned int output_dim;
	unsigned int kernel_size;
	unsigned int stride;
	unsigned int padding;
	
	tensor* kernel_w;
	tensor* kernel_b;
}CONV2D;

typedef struct POOL
{
	layer_type type;
	
	unsigned int polsize;
	
	tensor* mask;
	
}POOL;

typedef struct layer_list
{
	layer_type type;
	
	tensor* input;
	union data
	{
		FC* fclayer;
		CONV2D* conv2dlayer;
		POOL* poollayer;
	}data;
	tensor* output;
	
	active_func FUN;
	active_func_d dFUN;
	
	tensor* actioutput;
	
	struct layer_list* pre;
	struct layer_list* next;
}layer_list;

typedef struct net
{
	layer_list* layer_fount;
	layer_list* layer_tail;
	unsigned int layernum;
}net,*net_handle;

net_handle create_net();//新建一个网络
void add_fc_layer(net_handle NET,unsigned int inputnum,unsigned int outputnum,actifun_type actifun);//给网络增加全连接层
void add_conv2d_layer(net_handle NET,unsigned int outputdim,unsigned int kernelsize,unsigned int stride,unsigned int padding,actifun_type actifun);//给网络增加卷积层
tensor_handle fc_forward(FC* layer,tensor_handle input);//全连接层前向传播
void forward(net_handle NET,tensor_handle input);//网络前向传播
float compute_lose(net_handle NET,lose_func FUN,tensor_handle ture_val);//计算损失
tensor_handle fc_backward(FC* layer, tensor* output_grad, tensor* input);//全连接层反向传播
void backward(net_handle NET, lose_func_d FUN, tensor_handle tureval);//网络反向传播
void optimizer(net_handle NET, float lr);//优化器
void net_zero_grad(net_handle NET);//清零网络梯度

#endif 

