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

void add_conv2d_layer(net_handle NET,unsigned int Cin,unsigned int kernelnum,unsigned int kernelsize,unsigned int stride,unsigned int padding,actifun_type actifun)
{
	if(NET == NULL)
	{
		printf("net not init!(add_conv2d_layer)\r\n");
		return;
	}
	CONV2D* conv2dlayer = (CONV2D*)malloc(sizeof(CONV2D));
	layer_list* node = (layer_list*)malloc(sizeof(layer_list));
	conv2dlayer->type = conv2d;
	conv2dlayer->Cin = Cin;
	conv2dlayer->kernelnum = kernelnum;
	conv2dlayer->kernel_size = kernelsize;
	conv2dlayer->stride = stride;
	conv2dlayer->padding = padding;
	conv2dlayer->kernel_w = create_tensor(kernelnum,Cin,kernelsize*kernelsize,random,grad);
	conv2dlayer->kernel_b = NULL; 
	
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

void add_pool_layer(net_handle NET,unsigned int poolsize)
{
	if(NET == NULL)
	{
		printf("net not init!(add_pool_layer)\r\n");
		return;
	}
	POOL* poollayer = (POOL*)malloc(sizeof(POOL));
	layer_list* node = (layer_list*)malloc(sizeof(layer_list));
	poollayer->type = pool;
	poollayer->polsize = poolsize;
	poollayer->mask = NULL;
	
	node->data.poollayer = poollayer;
	node->input = NULL;
	node->output = NULL;
	node->actioutput = NULL;
	node->type = pool;
	node->FUN = NULL;
	node->dFUN = NULL;
	
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

void add_flatten_layer(net_handle NET)
{
	if(NET == NULL)
	{
		printf("net not init!(add_flatten_layer)\r\n");
		return;
	}
	FLATTEN* flattenlayer = (FLATTEN*)malloc(sizeof(FLATTEN));
	layer_list* node = (layer_list*)malloc(sizeof(layer_list));
	flattenlayer->type = flatten;
	
	node->data.flattenlayer = flattenlayer;
	node->input = NULL;
	node->output = NULL;
	node->actioutput = NULL;
	node->type = flatten;
	node->FUN = NULL;
	node->dFUN = NULL;
	
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
	unsigned int Cin = input->dim;
	unsigned int ksize = layer->kernel_size;
	unsigned int str = layer->stride;
	unsigned int pad = layer->padding;
	unsigned int Knum = layer->kernelnum;
	
	tensor_handle padded = tensor_padding(input, pad, zero);
	unsigned int p_h = padded->h;
	unsigned int p_w = padded->w;
	
	unsigned int out_h = (p_h - ksize) / str + 1;
	unsigned int out_w = (p_w - ksize) / str + 1;
	unsigned int out_dim = Knum;
	
	if(layer->kernel_b == NULL || layer->kernel_b->w != out_dim)
	{
		if(layer->kernel_b != NULL)
			layer->kernel_b = remove_tensor(layer->kernel_b);
		layer->kernel_b = create_tensor(1, 1, out_dim, zero, grad);
	}
	
	tensor_handle ret = create_tensor(out_dim, out_h, out_w, zero, nograd);
	
	for(unsigned int k = 0;k < Knum;k++) 
	{
		for(unsigned int j = 0;j < out_h;j++)
		{
			for(unsigned int i = 0;i < out_w;i++)
			{
				float sum = 0.0f;
				for(unsigned int c = 0;c < Cin;c++) 
				{
					for(unsigned int l = 0;l < ksize;l++)
					{
						for(unsigned int m = 0;m < ksize;m++)
						{
							unsigned int ph = j * str + l; 
							unsigned int pw = i * str + m;
							float in_val = padded->Data[c * p_h * p_w + ph * p_w + pw]; 
							float w_val = layer->kernel_w->Data[k * Cin * ksize * ksize + c * ksize * ksize + l * ksize + m]; 
							sum += in_val * w_val;
						}
					}
				}
				ret->Data[k * out_h * out_w + j * out_w + i] = sum + layer->kernel_b->Data[k]; 
			}
		}
	}
	
	remove_tensor(padded);
	return ret;
}

tensor_handle pool_forward(POOL* layer,tensor_handle input)
{
	unsigned int dim = input->dim;
	unsigned int in_h = input->h;
	unsigned int in_w = input->w;
	unsigned int ps = layer->polsize;
	
	unsigned int out_h = in_h / ps;
	unsigned int out_w = in_w / ps;
	
	tensor_handle ret = create_tensor(dim, out_h, out_w, zero, nograd);
	
	if(layer->mask != NULL)layer->mask = remove_tensor(layer->mask);
	layer->mask = create_tensor(dim, out_h, out_w, zero, nograd);
	
	for(unsigned int c = 0;c < dim;c++)
	{
		for(unsigned int j = 0;j < out_h;j++)
		{
			for(unsigned int i = 0;i < out_w;i++)
			{
				float maxval = input->Data[c*in_h*in_w + (j*ps)*in_w + (i*ps)]; 
				unsigned int max_idx = 0;
				for(unsigned int l = 0;l < ps;l++)
				{
					for(unsigned int m = 0;m < ps;m++)
					{
						float v = input->Data[c*in_h*in_w + (j*ps+l)*in_w + (i*ps+m)]; 
						if(v > maxval)
						{
							maxval = v;
							max_idx = l*ps + m;
						}
					}
				}
				ret->Data[c*out_h*out_w + j*out_w + i] = maxval; 
				layer->mask->Data[c*out_h*out_w + j*out_w + i] = (float)max_idx; 
			}
		}
	}
	
	return ret;
}

tensor_handle flatten_forward(tensor_handle input)
{
	return tensor_to_line(input);
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
			if(p->output!=NULL)p->output = remove_tensor(p->output);
			p->output = conv2d_forward(p->data.conv2dlayer, p->input);
			break;
		case pool:
			if(p->output!=NULL)p->output = remove_tensor(p->output);
			p->output = pool_forward(p->data.poollayer, p->input);
			break;
		case flatten:
			if(p->output!=NULL)p->output = remove_tensor(p->output);
			p->output = flatten_forward(p->input);
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

tensor_handle conv2d_backward(CONV2D* layer, tensor_handle delta, tensor_handle input)
{
	unsigned int Cin = input->dim;
	unsigned int in_h = input->h;
	unsigned int in_w = input->w;
	unsigned int ksize = layer->kernel_size;
	unsigned int str = layer->stride;
	unsigned int pad = layer->padding;
	unsigned int Knum = layer->kernelnum;
	
	unsigned int out_h = delta->h;
	unsigned int out_w = delta->w;
	
	tensor_handle padded = tensor_padding(input, pad, zero);
	unsigned int p_h = padded->h;
	unsigned int p_w = padded->w;
	
	tensor_handle padded_grad = create_tensor(Cin, p_h, p_w, zero, nograd);
	
	for(unsigned int k = 0;k < Knum;k++) 
	{
		for(unsigned int j = 0;j < out_h;j++)
		{
			for(unsigned int i = 0;i < out_w;i++)
			{
				float d = delta->Data[k * out_h * out_w + j * out_w + i];
				layer->kernel_b->grad[k] += d;
				for(unsigned int c = 0;c < Cin;c++)
				{
					for(unsigned int l = 0;l < ksize;l++)
					{
						for(unsigned int m = 0;m < ksize;m++)
						{
							unsigned int w_idx = k * Cin * ksize * ksize + c * ksize * ksize + l * ksize + m;
							unsigned int ph = j * str + l; 
							unsigned int pw = i * str + m;
							float in_val = padded->Data[c * p_h * p_w + ph * p_w + pw];
							
							layer->kernel_w->grad[w_idx] += in_val * d;
							
							padded_grad->Data[c * p_h * p_w + ph * p_w + pw] += layer->kernel_w->Data[w_idx] * d;
						}
					}
				}
			}
		}
	}
	
	tensor_handle input_grad = create_tensor(Cin, in_h, in_w, zero, nograd);
	for(unsigned int c = 0;c < Cin;c++)
	{
		for(unsigned int j = 0;j < in_h;j++)
		{
			for(unsigned int i = 0;i < in_w;i++)
			{
				input_grad->Data[c * in_h * in_w + j * in_w + i] = padded_grad->Data[c * p_h * p_w + (j + pad) * p_w + (i + pad)];
			}
		}
	}
	
	remove_tensor(padded);
	remove_tensor(padded_grad);
	
	return input_grad;
}

tensor_handle pool_backward(POOL* layer, tensor_handle delta, tensor_handle input)
{
	unsigned int dim = input->dim;
	unsigned int in_h = input->h;
	unsigned int in_w = input->w;
	unsigned int ps = layer->polsize;
	
	unsigned int out_h = delta->h;
	unsigned int out_w = delta->w;
	
	tensor_handle input_grad = create_tensor(dim, in_h, in_w, zero, nograd);
	
	if(layer->mask == NULL)return input_grad;
	
	for(unsigned int c = 0;c < dim;c++)
	{
		for(unsigned int j = 0;j < out_h;j++)
		{
			for(unsigned int i = 0;i < out_w;i++)
			{
				unsigned int idx = (unsigned int)layer->mask->Data[c*out_h*out_w + j*out_w + i];
				unsigned int l = idx / ps;
				unsigned int m = idx % ps;
				input_grad->Data[c*in_h*in_w + (j*ps+l)*in_w + (i*ps+m)] = delta->Data[c*out_h*out_w + j*out_w + i]; 
			}
		}
	}
	
	return input_grad;
}

tensor_handle flatten_backward(tensor_handle delta, tensor_handle input)
{
	unsigned int datanum = input->w * input->h * input->dim;
	tensor_handle ret = create_tensor(input->dim, input->h, input->w, zero, nograd);
	for(unsigned int i = 0;i < datanum;i++)
	{
		ret->Data[i] = delta->Data[i];
	}
	return ret;
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
			{
				tensor* pre_delta = conv2d_backward(p->data.conv2dlayer, delta, input);
				remove_tensor(delta);
				delta = pre_delta;
				break;
			}
		case pool:
			{
				tensor* pre_delta = pool_backward(p->data.poollayer, delta, input);
				remove_tensor(delta);
				delta = pre_delta;
				break;
			}
		case flatten:
			{
				tensor* pre_delta = flatten_backward(delta, input);
				remove_tensor(delta);
				delta = pre_delta;
				break;
			}
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
			{
				CONV2D* conv = p->data.conv2dlayer;
				
				tensor* wgrad = grad_to_tensor(conv->kernel_w);
				tensor* w_lr = tensor_mul_k(wgrad, lr);
				tensor* new_w = tensor_per_val_sub(conv->kernel_w, w_lr);
				change_grad_mode(new_w, grad);
				
				remove_tensor(conv->kernel_w);
				conv->kernel_w = new_w;
				
				remove_tensor(wgrad);
				remove_tensor(w_lr);
				
				tensor* bgrad = grad_to_tensor(conv->kernel_b);
				tensor* b_lr = tensor_mul_k(bgrad, lr);
				tensor* new_b = tensor_per_val_sub(conv->kernel_b, b_lr);
				change_grad_mode(new_b, grad);
				
				remove_tensor(conv->kernel_b);
				conv->kernel_b = new_b;
				
				remove_tensor(bgrad);
				remove_tensor(b_lr);
				
				
				break;
			}
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
