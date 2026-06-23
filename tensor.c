#include "tensor.h"

tensor_handle create_tensor(unsigned int dim,unsigned int h,unsigned int w,create_tensor_mode mode,isgrad gradmode)
{
	unsigned int i = 0,j = 0,k = 0;
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	ret->w = w;
	ret->h = h;
	ret->dim = dim;
	
	ret->Data = (tensor_data_type*)malloc((w*h*dim)*sizeof(tensor_data_type));
	if(gradmode == grad)
	{
		ret->grad = (tensor_data_type*)malloc((w*h*dim)*sizeof(tensor_data_type));
		for(i = 0;i<(w*h*dim);i++)
		{
			ret->grad[i] = 0.0;
		}
	}else
	{
		ret->grad = NULL;
	}
	
	ret->req_grad = gradmode;
	
	switch (mode)
	{
		case random:
			srand((unsigned int)time(NULL));
			for(i = 0;i<(w*h*dim);i++)
			{
				ret->Data[i] = (float)(((float)rand()/(float)RAND_MAX)*0.01);
			}
			break;
		case zero:
			for(i = 0;i<(w*h*dim);i++)
			{
				ret->Data[i] = 0;
			}
			break;
		case one:
			for(i = 0;i<(w*h*dim);i++)
			{
				ret->Data[i] = 1;
			}
			break;
		case eye:
			for(i = 0;i<dim;i++)
			{
				for(j = 0;j<h;j++)
				{
					for(k = 0;k<w;k++)
					{
						if(j == k)ret->Data[(i*w*h)+(j*w)+k] = 1;
						else ret->Data[(i*w*h)+(j*w)+k] = 0;
					}
				}
			}
			break;
		default:
			break;
	}
	return ret;
}

tensor_handle create_tensor_from_twodim_arr(unsigned int h,unsigned int w,isgrad gradmode,void* arr)
{
	tensor_data_type(*p)[w] = (tensor_data_type(*)[w])arr;
	unsigned int i = 0,j = 0,k = 0;
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	ret->w = w;
	ret->h = h;
	ret->dim = 1;
	
	ret->Data = (tensor_data_type*)malloc((w*h)*sizeof(tensor_data_type));
	if(gradmode == grad)
	{
		ret->grad = (tensor_data_type*)malloc((w*h)*sizeof(tensor_data_type));
		for(i = 0;i<(w*h);i++)
		{
			ret->grad[i] = 0.0;
		}
	}else
	{
		ret->grad = NULL;
	}
	
	ret->req_grad = gradmode;
	for(i = 0;i<1;i++)
	{
		for(j = 0;j<h;j++)
		{
			for(k = 0;k<w;k++)
			{
				ret->Data[(i*w*h)+(j*w)+k] = p[j][k];
			}
		}
	}
	return ret;
}

tensor_handle remove_tensor(tensor_handle tensortoremove)
{
	if(tensortoremove == NULL)return NULL;
	free(tensortoremove->Data);
	free(tensortoremove->grad);
	free(tensortoremove);
	tensortoremove = NULL;
	return NULL;
}

tensor_data_type get_tensor_data(tensor_handle tensortoshow,unsigned int dim,unsigned int h,unsigned int w)
{
	return tensortoshow->Data[((dim-1)*(tensortoshow->w)*(tensortoshow->h))+((h-1)*tensortoshow->w)+(w-1)];
}

tensor_handle tensor_T(tensor_handle tensortoT)
{
	unsigned int i = 0,j = 0,k = 0;
	unsigned int datanum = (tensortoT->w)*(tensortoT->h)*(tensortoT->dim);
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	
	ret->dim = tensortoT->dim;
	ret->h = tensortoT->w;
	ret->w = tensortoT->h;
	ret->req_grad = nograd;
	ret->Data = (tensor_data_type*)malloc(datanum*sizeof(tensor_data_type));
	ret->grad = NULL;

	
	for(i = 0;i<ret->dim;i++)
	{
		for(j = 0;j<ret->h;j++)
		{
			for(k = 0;k<ret->w;k++)
			{
				ret->Data[(i*(ret->w)*(ret->h))+(j*(ret->w))+k] = tensortoT->Data[(i*(tensortoT->w)*(tensortoT->h))+(k*(tensortoT->w))+j];
			}
		}
	}
	return ret;
}

tensor_handle tensor_matrix_mul(tensor_handle tensorA, tensor_handle tensorB)
{
	if (tensorA == NULL || tensorB == NULL)
	{
		printf("tensor is not init!(tensor_matrix_mul)\n");
		return NULL;
	}
	
	if (tensorA->dim != 1 || tensorB->dim != 1)
	{
		printf("tensor can not mul! only support 2D tensor (tensor_matrix_mul)\n");
		return NULL;
	}
	
	if (tensorA->w != tensorB->h)
	{
		printf("tensor can not mul! A.w != B.h(tensor_matrix_mul)\n");
		return NULL;
	}
	
	unsigned int ret_h = tensorA->h;
	unsigned int ret_w = tensorB->w;
	unsigned int data_num = ret_h * ret_w;
	
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	if (!ret) { printf("malloc failed(tensor_matrix_mul)\n"); return NULL; }
	
	ret->dim = 1;
	ret->h = ret_h;
	ret->w = ret_w;
	ret->req_grad = nograd;
	ret->Data = (tensor_data_type*)malloc(data_num * sizeof(tensor_data_type));
	ret->grad = NULL;
	
	for (unsigned int j = 0; j < ret_h; j++)
	{
		for (unsigned int k = 0; k < ret_w; k++)
		{
			tensor_data_type sum = 0.0;
			for (unsigned int l = 0; l < tensorA->w; l++)
			{
				tensor_data_type a = tensorA->Data[j * tensorA->w + l];
				tensor_data_type b = tensorB->Data[l * tensorB->w + k];
				sum += a * b;
			}
			ret->Data[j * ret_w + k] = sum;
		}
	}
	
	return ret;
}

tensor_handle tensor_per_val_mul(tensor_handle tensorA,tensor_handle tensorB)
{
	if(tensorA == NULL || tensorB == NULL)
	{
		printf("tensor is not init!(tensor_per_val_mul)\r\n");
		return NULL;
	}
	if(tensorA->dim != tensorB->dim || tensorA->h != tensorB->h || tensorA->w != tensorB->w)
	{
		printf("do per val mul,the two tensor must have same shape(tensor_per_val_mul)\r\n");
		return NULL;
	}
	unsigned int i = 0;
	unsigned int datanum = (tensorA->w)*(tensorA->h)*(tensorA->dim);
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	ret->dim = tensorA->dim;
	ret->h = tensorA->h;
	ret->w = tensorA->w;
	ret->req_grad = nograd;
	ret->Data = (tensor_data_type*)malloc(datanum*sizeof(tensor_data_type));
	ret->grad = NULL;
	for(i = 0;i<datanum;i++)
	{
		ret->Data[i] = (tensorA->Data[i])*(tensorB->Data[i]);
	}
	return ret;
}

tensor_handle tensor_mul_k(tensor_handle tensorA,tensor_data_type k)
{
	if(tensorA == NULL)
	{
		printf("tensor is not init!(tensor_mul_k)\r\n");
		return NULL;
	}
	unsigned int i = 0;
	unsigned int datanum = (tensorA->w)*(tensorA->h)*(tensorA->dim);
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	ret->dim = tensorA->dim;
	ret->h = tensorA->h;
	ret->w = tensorA->w;
	ret->req_grad = nograd;
	ret->Data = (tensor_data_type*)malloc(datanum*sizeof(tensor_data_type));
	ret->grad = NULL;
	for(i = 0;i<datanum;i++)
	{
		ret->Data[i] = (tensorA->Data[i])*k;
	}
	return ret;
}

tensor_handle tensor_sum_all(tensor_handle tensorA)
{
	if(tensorA == NULL)
	{
		printf("tensor is not init!(tensor_sum_all)\r\n");
		return NULL;
	}
	unsigned int i = 0;
	unsigned int datanum = (tensorA->w)*(tensorA->h)*(tensorA->dim);
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	ret->dim = 1;
	ret->h = 1;
	ret->w = 1;
	ret->req_grad = nograd;
	ret->Data = (tensor_data_type*)malloc(sizeof(tensor_data_type));
	ret->grad = NULL;
	
	ret->Data[0] = 0.0;
	
	for(i = 0;i<datanum;i++)
	{
		ret->Data[0] += (tensorA->Data[i]);
	}
	
	return ret;
}

tensor_handle tensor_per_val_sum(tensor_handle tensorA,tensor_handle tensorB)
{
	if(tensorA == NULL || tensorB == NULL)
	{
		printf("tensor is not init!(tensor_per_val_sum)\r\n");
		return NULL;
	}
	if(tensorA->dim != tensorB->dim || tensorA->h != tensorB->h || tensorA->w != tensorB->w)
	{
		printf("do per val sum,the two tensor must have same shape(tensor_per_val_sum)\r\n");
		return NULL;
	}
	unsigned int i = 0;
	unsigned int datanum = (tensorA->w)*(tensorA->h)*(tensorA->dim);
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	ret->dim = tensorA->dim;
	ret->h = tensorA->h;
	ret->w = tensorA->w;
	ret->req_grad = nograd;
	ret->Data = (tensor_data_type*)malloc(datanum*sizeof(tensor_data_type));
	ret->grad = NULL;
	for(i = 0;i<datanum;i++)
	{
		ret->Data[i] = (tensorA->Data[i])+(tensorB->Data[i]);
	}
	return ret;
}

tensor_handle tensor_per_val_sub(tensor_handle tensorA,tensor_handle tensorB)
{
	if(tensorA == NULL || tensorB == NULL)
	{
		printf("tensor is not init!(tensor_per_val_sub)\r\n");
		return NULL;
	}
	if(tensorA->dim != tensorB->dim || tensorA->h != tensorB->h || tensorA->w != tensorB->w)
	{
		printf("do per val sub,the two tensor must have same shape(tensor_per_val_sub)\r\n");
		return NULL;
	}
	unsigned int i = 0;
	unsigned int datanum = (tensorA->w)*(tensorA->h)*(tensorA->dim);
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	ret->dim = tensorA->dim;
	ret->h = tensorA->h;
	ret->w = tensorA->w;
	ret->req_grad = nograd;
	ret->Data = (tensor_data_type*)malloc(datanum*sizeof(tensor_data_type));
	ret->grad = NULL;
	for(i = 0;i<datanum;i++)
	{
		ret->Data[i] = (tensorA->Data[i])-(tensorB->Data[i]);
	}
	return ret;
}

tensor_handle tensor_to_line(tensor_handle tensortoline)
{
	int i = 0;
	unsigned int datanum = (tensortoline->w)*(tensortoline->h)*(tensortoline->dim);
	tensor* ret = (tensor*)malloc(sizeof(tensor));
	
	ret->dim = 1;
	ret->h = 1;
	ret->w = datanum;
	ret->req_grad = nograd;
	ret->Data = (tensor_data_type*)malloc(datanum*sizeof(tensor_data_type));
	ret->grad = NULL;
	
	for(i = 0;i<datanum;i++)
	{
		ret->Data[i] = tensortoline->Data[i];
	}
	return ret;
}

tensor_handle tensor_padding(tensor_handle tensorpad,unsigned int size,create_tensor_mode mode)
{
	unsigned int i = 0,j = 0,k = 0;
	unsigned int dim = tensorpad->dim;
	unsigned int h = tensorpad->h;
	unsigned int w = tensorpad->w;
	
	tensor* ret = create_tensor(tensorpad->dim,(tensorpad->h+size*2),(tensorpad->w+size*2),mode,nograd);
	for(i = 0;i<dim;i++)
	{
		for(j = 0;j<h;j++)
		{
			for(k = 0;k<w;k++)
			{
				ret->Data[((i*(h+2*size)*(w+2*size))+((j+1)*(w+2*size))+(k+1))] = tensorpad->Data[((i*h*w)+(j*w)+k)];
			}
		}
	}
	return ret;
}

void tensor_zero_grad(tensor_handle tensortozerograd)
{
	if(tensortozerograd->req_grad == nograd)return;
	unsigned int datanum = (tensortozerograd->w)*(tensortozerograd->h)*(tensortozerograd->dim);
	unsigned int i = 0;
	for(i = 0;i<datanum;i++)
	{
		tensortozerograd->grad[i] = 0.0;
	}
}
void change_grad_mode(tensor_handle tensortochangegrad,isgrad gradmode)
{
	int i = 0;
	unsigned int datanum = (tensortochangegrad->w)*(tensortochangegrad->h)*(tensortochangegrad->dim);
	if(gradmode == grad)
	{
		if(tensortochangegrad->grad == NULL)
		{
			tensortochangegrad->grad = (tensor_data_type*)malloc(datanum*sizeof(tensor_data_type));
			for(i = 0;i<datanum;i++)
			{
				tensortochangegrad->grad[i] = 0.0;
			}
		}
		tensortochangegrad->req_grad = grad;
		return;
	}else
	{
		if(tensortochangegrad->grad != NULL)
		{
			free(tensortochangegrad->grad);
			tensortochangegrad->grad = NULL;
		}
		tensortochangegrad->req_grad = nograd;
		return;
	}
}

void tensor_to_grad(tensor_handle tensorupdatagrad,tensor_handle gradtensor)
{
	if(tensorupdatagrad == NULL || gradtensor == NULL)return;
	if(tensorupdatagrad->req_grad == nograd)return;
	if(tensorupdatagrad->dim!=gradtensor->dim || tensorupdatagrad->h!=gradtensor->h || tensorupdatagrad->w!=gradtensor->w)
	{
		printf("do grad update,the two tensor must have same shape(tensor_to_grad)\r\n");
		return;
	}
	unsigned int datanum = (tensorupdatagrad->dim)*(tensorupdatagrad->h)*(tensorupdatagrad->w);
	int i = 0;
	for(i = 0;i<datanum;i++)
	{
		tensorupdatagrad->grad[i] = gradtensor->Data[i];
	}
	return;
}

void tensor_sum_to_grad(tensor_handle tensorupdatagrad,tensor_handle gradtensor)
{
	if(tensorupdatagrad == NULL || gradtensor == NULL)return;
	if(tensorupdatagrad->req_grad == nograd)return;
	if(tensorupdatagrad->dim!=gradtensor->dim || tensorupdatagrad->h!=gradtensor->h || tensorupdatagrad->w!=gradtensor->w)
	{
		printf("do grad update,the two tensor must have same shape(tensor_sum_to_grad)\r\n");
		return;
	}
	unsigned int datanum = (tensorupdatagrad->dim)*(tensorupdatagrad->h)*(tensorupdatagrad->w);
	int i = 0;
	for(i = 0;i<datanum;i++)
	{
		tensorupdatagrad->grad[i] += gradtensor->Data[i];
	}
	return;
}

tensor_handle grad_to_tensor(tensor_handle tensorA)
{
	if(tensorA == NULL)
	{
		printf("tensor is not init!(grad_to_tensor)\r\n");
		return NULL;
	}
	if(tensorA->req_grad == nograd)
	{
		printf("tensor is nograd!%d(grad_to_tensor)\r\n",tensorA->req_grad);
		return NULL;
	}
	
	tensor* ret  = create_tensor(tensorA->dim,tensorA->h,tensorA->w,zero,nograd);
	unsigned int datanum = (tensorA->dim)*(tensorA->h)*(tensorA->w);
	int i = 0;
	for(i = 0;i<datanum;i++)
	{
		ret->Data[i] = tensorA->grad[i];
	}
	return ret;
}

void show_tensor(tensor_handle tensortoshow)
{
	if(tensortoshow == NULL)return;
	unsigned int i = 0,j = 0,k = 0;
	unsigned int w = tensortoshow->w;
	unsigned int h = tensortoshow->h;
	unsigned int dim = tensortoshow->dim;
	printf("tensor %dx%dx%d,grad req = %d\r\n",dim,h,w,tensortoshow->req_grad);
	printf("{\r\n");
	for(i = 0;i<dim;i++)
	{
		printf("[\r\n");
		for(j = 0;j<h;j++)
		{
			printf("(");
			for(k = 0;k<w;k++)
			{
				printf("%.4f    ",tensortoshow->Data[(i*w*h)+(j*w)+k]);
			}
			printf(")\r\n");
		}
		printf("]\r\n");
	}
	printf("}\r\n");
}

void show_tensor_grad(tensor_handle tensortoshow)
{
	if(tensortoshow == NULL)return;
	if(tensortoshow->req_grad == nograd)return;
	unsigned int i = 0,j = 0,k = 0;
	unsigned int w = tensortoshow->w;
	unsigned int h = tensortoshow->h;
	unsigned int dim = tensortoshow->dim;
	printf("tensor %dx%dx%d\r\n",dim,h,w);
	printf("{\r\n");
	for(i = 0;i<dim;i++)
	{
		printf("[\r\n");
		for(j = 0;j<h;j++)
		{
			printf("(");
			for(k = 0;k<w;k++)
			{
				printf("%.4f    ",tensortoshow->grad[(i*w*h)+(j*w)+k]);
			}
			printf(")\r\n");
		}
		printf("]\r\n");
	}
	printf("}\r\n");
}

