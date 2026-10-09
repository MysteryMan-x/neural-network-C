#include "net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================= 可调参数 ================= */
#define EPOCHS             2        /* 训练轮数 */
#define LR                 0.01f    /* 学习率 */
#define TRAIN_PRINT_EVERY  100     /* 训练时每多少个样本打印一次进度 */
#define IMG_ROWS           28
#define IMG_COLS           28
#define IMG_SIZE           (IMG_ROWS * IMG_COLS)
#define N_CLASSES          10

/* ================= 工具函数 ================= */

/* 从二进制文件读取一个大端序的 32 位无符号整数 */
static unsigned int read_be32(FILE* f)
{
	unsigned char b[4];
	if(fread(b, 1, 4, f) != 4) return 0;
	return ((unsigned int)b[0] << 24) | ((unsigned int)b[1] << 16) |
	((unsigned int)b[2] << 8)  |  (unsigned int)b[3];
}

/* 读取 MNIST 图像文件 (IDX 格式, magic=2051) */
static int load_images(const char* path, unsigned char** out_pixels,
	unsigned int* out_count, unsigned int* out_rows, unsigned int* out_cols)
{
	FILE* f = fopen(path, "rb");
	if(f == NULL)
	{
		printf("[错误] 无法打开图像文件: %s\n", path);
		return -1;
	}
	unsigned int magic = read_be32(f);
	unsigned int count = read_be32(f);
	unsigned int rows  = read_be32(f);
	unsigned int cols  = read_be32(f);
	if(magic != 2051)
	{
		printf("[错误] %s 不是 MNIST 图像文件 (magic=%u)\n", path, magic);
		fclose(f);
		return -2;
	}
	unsigned int n = count * rows * cols;
	unsigned char* buf = (unsigned char*)malloc(n);
	if(buf == NULL || fread(buf, 1, n, f) != n)
	{
		printf("[错误] 读取图像数据失败: %s\n", path);
		if(buf != NULL) free(buf);
		fclose(f);
		return -3;
	}
	fclose(f);
	*out_pixels = buf;
	*out_count  = count;
	*out_rows   = rows;
	*out_cols   = cols;
	return 0;
}

/* 读取 MNIST 标签文件 (IDX 格式, magic=2049) */
static int load_labels(const char* path, unsigned char** out_labels, unsigned int* out_count)
{
	FILE* f = fopen(path, "rb");
	if(f == NULL)
	{
		printf("[错误] 无法打开标签文件: %s\n", path);
		return -1;
	}
	unsigned int magic = read_be32(f);
	unsigned int count = read_be32(f);
	if(magic != 2049)
	{
		printf("[错误] %s 不是 MNIST 标签文件 (magic=%u)\n", path, magic);
		fclose(f);
		return -2;
	}
	unsigned char* buf = (unsigned char*)malloc(count);
	if(buf == NULL || fread(buf, 1, count, f) != count)
	{
		printf("[错误] 读取标签数据失败: %s\n", path);
		if(buf != NULL) free(buf);
		fclose(f);
		return -3;
	}
	fclose(f);
	*out_labels = buf;
	*out_count  = count;
	return 0;
}

/* 返回数组中最大值的下标 */
static int argmax(const float* data, unsigned int n)
{
	unsigned int best = 0;
	unsigned int i;
	for(i = 1; i < n; i++)
		if(data[i] > data[best]) best = i;
	return (int)best;
}

/* 把一张图的像素写入输入张量(归一化到 [0,1]) */
static void fill_input(tensor_handle x, const unsigned char* img)
{
	unsigned int p;
	for(p = 0; p < IMG_SIZE; p++)
		x->Data[p] = (float)img[p] / 255.0f;
}

/* 把标签写入 one-hot 目标张量 */
static void fill_target(tensor_handle y, unsigned char label)
{
	memset(y->Data, 0, N_CLASSES * sizeof(float));
	y->Data[label] = 1.0f;
}

/* ================= 主函数 ================= */
int main(void)
{
	unsigned char *train_img = NULL, *train_lbl = NULL;
	unsigned char *test_img  = NULL, *test_lbl  = NULL;
	unsigned int train_n = 0, test_n = 0, rows = 0, cols = 0;
	
	printf("==== 读取 MNIST 数据集 ====\n");
	if(load_images("train-images-idx3-ubyte", &train_img, &train_n, &rows, &cols) != 0) return 1;
	if(load_labels("train-labels-idx1-ubyte", &train_lbl, &train_n) != 0) return 1;
	if(load_images("t10k-images-idx3-ubyte", &test_img, &test_n, &rows, &cols) != 0) return 1;
	if(load_labels("t10k-labels-idx1-ubyte", &test_lbl, &test_n) != 0) return 1;
	
	if(rows != IMG_ROWS || cols != IMG_COLS)
	{
		printf("[错误] 图像尺寸是 %ux%u, 本程序按 28x28 处理\n", rows, cols);
		return 1;
	}
	printf("训练集 %u 张, 测试集 %u 张, 图像 %ux%u\n", train_n, test_n, rows, cols);
	
	printf("\n==== 构建网络 ====\n");
	net_handle net = create_net();
	add_conv2d_layer(net, 1, 64, 3, 1, 1, relu);   /* 28x28x1 -> 28x28x64  */
	add_pool_layer(net, 2);                     /* -> 14x14x64         */
	add_conv2d_layer(net, 64, 32, 3, 1, 1, relu);  /* -> 14x14x32         */
	add_pool_layer(net, 2);                     /* -> 7x7x32         */
	add_flatten_layer(net);                     /* -> 1568            */
	add_fc_layer(net, 7*7*32, 512, relu);       /* -> 512              */
	add_fc_layer(net, 512, 64, relu);          /* -> 64              */
	add_fc_layer(net, 64, 10, none);           /* -> 10              */
	printf("网络构建完成, 共 %u 层\n", net->layernum);
	
	/* 复用输入/目标张量, 避免每次训练都重新 malloc */
	tensor_handle x = create_tensor(1, IMG_ROWS, IMG_COLS, zero, nograd);
	tensor_handle y = create_tensor(1, 1, N_CLASSES, zero, nograd);
	
	printf("\n==== 开始训练 (epoch=%d, lr=%.4f) ====\n", EPOCHS, LR);
	int ep;
	for(ep = 0; ep < EPOCHS; ep++)
	{
		float loss_sum = 0.0f;
		unsigned int correct = 0;
		unsigned int i;
		
		for(i = 0; i < train_n; i++)
		{
			fill_input(x, train_img + (size_t)i * IMG_SIZE);
			fill_target(y, train_lbl[i]);
			
			forward(net, x);
			loss_sum += compute_lose(net, MSE, y);
			if(argmax(net->layer_tail->output->Data, N_CLASSES) == (int)train_lbl[i])
				correct++;
			backward(net, MSE_d, y);
			optimizer(net, LR);
			
			if((i + 1) % TRAIN_PRINT_EVERY == 0)
			{
				printf("  epoch %d/%d, 样本 %u/%u, 平均loss=%.4f, 训练准确率=%.2f%%\n",
					ep + 1, EPOCHS, i + 1, train_n,
					loss_sum / (float)(i + 1),
					100.0f * (float)correct / (float)(i + 1));
			}
		}
		printf("epoch %d/%d 完成: 平均loss=%.4f, 训练准确率=%.2f%%\n",
			ep + 1, EPOCHS,
			loss_sum / (float)train_n,
			100.0f * (float)correct / (float)train_n);
	}
	
	printf("\n==== 测试集评估 ====\n");
	unsigned int correct = 0;
	unsigned int i;
	for(i = 0; i < test_n; i++)
	{
		fill_input(x, test_img + (size_t)i * IMG_SIZE);
		forward(net, x);
		if(argmax(net->layer_tail->output->Data, N_CLASSES) == (int)test_lbl[i])
			correct++;
	}
	printf("测试准确率: %u / %u = %.2f%%\n",
		correct, test_n, 100.0f * (float)correct / (float)test_n);
	
	/* 释放数据缓冲 */
	free(train_img);
	free(train_lbl);
	free(test_img);
	free(test_lbl);
	remove_tensor(x);
	remove_tensor(y);
	
	printf("\n训练测试结束。\n");
	return 0;
}
