#include "net.h"

net_handle create_net()
{
	net* ret = (net*)malloc(sizeof(net));
	ret->layer_fount = NULL;
	ret->layer_tail = NULL;
	ret->layernum = 0;
	return ret;
}

void add_fc_layer(net_handle NET,unsigned int inputnum,unsigned int outputnum,actifun_type actifun)
{
	if(NET == NULL)
	{
		printf("net not init!(add_fc_layer)\r\n");
		return;
	}
	FC* fclayer = (FC*)malloc(sizeof(FC));
	layer_list* node = (layer_list*)malloc(sizeof(layer_list));
	fclayer->type = fc;
	fclayer->input_node_num = inputnum;
	fclayer->output_node_num = outputnum;
	fclayer->layer_w_tensor = create_tensor(1,inputnum,outputnum,random,grad);
	fclayer->layer_b_tensor = create_tensor(1,1,outputnum,random,grad);
	
	node->data.fclayer = fclayer;
	node->input = NULL;
	node->output = NULL;
	node->actioutput = NULL;
	node->type = fc;
	switch (actifun)
	{
		case relu:
			node->FUN = tensor_relu;
			node->dFUN = tensor_relu_grad;
			break;
		case sigmoid:
			node->FUN = tensor_sigmoid;
			node->dFUN = tensor_sigmoid_grad;
			break;
		case none:
			node->FUN = NULL;
			node->dFUN = NULL;
			break;
		default:
			node->FUN = NULL;
			node->dFUN = NULL;
			break;
	}
	if(NET->layer_fount == NULL)
	{
		NET->layer_fount = node;
		NET->layer_tail = node;
		node->pre = NULL;
		node->next = NULL;
		NET->layernum++;
		return;
	}
	node->next = NULL;
	node->pre = NET->layer_tail;
	NET->layer_tail->next = node;
	NET->layer_tail = node;
	NET->layernum++;
	return;
}

void add_conv2d_layer(net_handle NET,unsigned int outputdim,unsigned int kernelsize,unsigned int stride,unsigned int padding,actifun_type actifun)
{
	if(NET == NULL)
	{
		printf("net not init!(add_conv2d_layer)\r\n");
		return;
	}
	CONV2D* conv2dlayer = (CONV2D*)malloc(sizeof(CONV2D));
	layer_list* node = (layer_list*)malloc(sizeof(layer_list));
	conv2dlayer->type = conv2d;
	conv2dlayer->output_dim = outputdim;
	conv2dlayer->kernel_size = kernelsize;
	conv2dlayer->stride = stride;
	conv2dlayer->padding = padding;
	conv2dlayer->kernel_w = create_tensor(outputdim,kernelsize,kernelsize,random,grad);
	conv2dlayer->kernel_b = create_tensor(1,1,outputdim,random,grad);
	
	node->data.conv2dlayer = conv2dlayer;
	node->input = NULL;
	node->output = NULL;
	node->actioutput = NULL;
	node->type = conv2d;
	switch (actifun)
	{
	case relu:
		node->FUN = tensor_relu;
		node->dFUN = tensor_relu_grad;
		break;
	case sigmoid:
		node->FUN = tensor_sigmoid;
		node->dFUN = tensor_sigmoid_grad;
		break;
	case none:
		node->FUN = NULL;
		node->dFUN = NULL;
		break;
	default:
		node->FUN = NULL;
		node->dFUN = NULL;
		break;
	}
	if(NET->layer_fount == NULL)
	{
		NET->layer_fount = node;
		NET->layer_tail = node;
		node->pre = NULL;
		node->next = NULL;
		NET->layernum++;
		return;
	}
	node->next = NULL;
	node->pre = NET->layer_tail;
	NET->layer_tail->next = node;
	NET->layer_tail = node;
	NET->layernum++;
	return;
}

tensor_handle fc_forward(FC* layer,tensor_handle input)
{
	tensor* mid = tensor_matrix_mul(input,layer->layer_w_tensor);
	tensor* ret = tensor_per_val_sum(mid,layer->layer_b_tensor);
	remove_tensor(mid);
	return ret;
}

tensor_handle conv2d_forward(CONV2D* layer,tensor_handle input)
{
	
	return NULL;
}

void forward(net_handle NET,tensor_handle input)
{
	if(NET == NULL)
	{
		printf("net not init!(forward)\r\n");
		return;
	}
	layer_list* p = NET->layer_fount;
	while(p!=NULL)
	{
		p->input = p->pre == NULL?input:p->pre->FUN!=NULL?p->pre->actioutput:p->pre->output;
		switch (p->type) {
		case fc:
			if(p->output!=NULL)p->output = remove_tensor(p->output);
			p->output = fc_forward(p->data.fclayer,p->input);
			break;
		case conv2d:
			//TODO
			break;
		default:
			//TODO
			break;
		}
		if(p->FUN != NULL)
		{
			if(p->actioutput != NULL)p->actioutput = remove_tensor(p->actioutput);
			p->actioutput = p->FUN(p->output);
		}
		p = p->next;
	}
}

float compute_lose(net_handle NET,lose_func FUN,tensor_handle ture_val)
{
	if(NET == NULL)
	{
		printf("net not init!(compute_lose)\r\n");
		return -1.0;
	}
	float lose = 0.0;
	lose = FUN(NET->layer_tail->actioutput ? NET->layer_tail->actioutput : NET->layer_tail->output, ture_val);
	return lose;
}

tensor_handle fc_backward(FC* layer, tensor* output_grad, tensor* input)
{
	tensor* input_T = tensor_T(input);
	tensor* w_grad = tensor_matrix_mul(input_T, output_grad);
	
	tensor_to_grad(layer->layer_w_tensor,w_grad);
	tensor_to_grad(layer->layer_b_tensor,output_grad);
	
	tensor* w_T = tensor_T(layer->layer_w_tensor);
	tensor* input_grad = tensor_matrix_mul(output_grad, w_T);
	
	remove_tensor(input_T);
	remove_tensor(w_grad);
	remove_tensor(w_T);
	
	return input_grad;
}

void backward(net_handle NET, lose_func_d FUN, tensor_handle tureval)
{
	if(NET == NULL)
	{
		printf("net not init!(backward)\r\n");
		return;
	}
	
	layer_list* p = NET->layer_tail;
	tensor* delta = NULL;
	
	delta = FUN(NET->layer_tail->actioutput ? NET->layer_tail->actioutput : NET->layer_tail->output, tureval);
	
	while(p != NULL)
	{
		tensor* input = NULL;
		if(p->pre == NULL)
		{
			input = p->input;
		}
		else
		{
			input = p->pre->actioutput ? p->pre->actioutput : p->pre->output;
		}
		
		if(p->dFUN != NULL)
		{
			tensor* tmp = delta;
			delta = p->dFUN(tmp, p->output);
			remove_tensor(tmp);
		}
		
		switch(p->type)
		{
		case fc:
			{
				tensor* pre_delta = fc_backward(p->data.fclayer, delta, input);
				remove_tensor(delta);
				delta = pre_delta;
				break;
			}
		case conv2d:
			break;
		default:
			break;
		}
		
		p = p->pre;
	}
	
	remove_tensor(delta);
}

void optimizer(net_handle NET, float lr)
{
	if(NET == NULL)
	{
		printf("net not init!(optimizer)\r\n");
		return;
	}
	
	layer_list* p = NET->layer_fount;
	while(p != NULL)
	{
		switch(p->type)
		{
		case fc:
			{
				FC* fc = p->data.fclayer;

				tensor* wgrad = grad_to_tensor(fc->layer_w_tensor);
				tensor* w_lr = tensor_mul_k(wgrad, lr);
				tensor* new_w = tensor_per_val_sub(fc->layer_w_tensor, w_lr);
				change_grad_mode(new_w,grad);
				remove_tensor(fc->layer_w_tensor);
				fc->layer_w_tensor = new_w;
				
				tensor* bgrad = grad_to_tensor(fc->layer_b_tensor);
				tensor* b_lr = tensor_mul_k(bgrad, lr);
				tensor* new_b = tensor_per_val_sub(fc->layer_b_tensor, b_lr);
				change_grad_mode(new_b,grad);
				remove_tensor(fc->layer_b_tensor);
				fc->layer_b_tensor = new_b;
				
				remove_tensor(wgrad);
				remove_tensor(bgrad);
				remove_tensor(w_lr);
				remove_tensor(b_lr);
				break;
			}
		case conv2d:
			break;
		default:
			break;
		}
		p = p->next;
	}
}

void net_zero_grad(net_handle NET)
{
	if(NET == NULL)
	{
		printf("net not init!(net_zero_grad)\r\n");
		return;
	}
	layer_list* p = NET->layer_fount;
	while(p!=NULL)
	{
		tensor_zero_grad(p->data.fclayer->layer_w_tensor);
		tensor_zero_grad(p->data.fclayer->layer_b_tensor);
		p = p->next;
	}
}

