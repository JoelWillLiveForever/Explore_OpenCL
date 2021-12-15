//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define _CRT_SECURE_NO_WARNINGS
//#define _CRT_NONSTDC_NO_DEPRECATE

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SRC_SIZE (0x100000)	// максимальный размер исходного кода kernel'а

int main()
{
	// создание массивов-векторов
	int i;
	const int arr_size = 100000;

	int* A = (int*)malloc(sizeof(int) * arr_size);
	int* B = (int*)malloc(sizeof(int) * arr_size);

	for (i = 0; i < arr_size; i++)
	{
		A[i] = i;
		B[i] = arr_size - i;
		/*printf("A = %d, B = %d\n", A[i], B[i]);*/
	}

	// загрузка исходного кода kernel'а
	int fd;
	char* src_str;
	size_t src_size;

	src_str = (char*)malloc(MAX_SRC_SIZE);
	fd = _open("kernel.cl", _O_RDONLY);
	if (fd <= 0)
	{
		fprintf(stderr, "Не получилось считать исходный файл kernel'а!");
		exit(1);
	}

	src_size = _read(fd, src_str, MAX_SRC_SIZE);
	_close(fd);

	// сбор информации о платформах и вычислительных девайсах
	cl_platform_id	platform_id = NULL;
	cl_device_id	device_id = NULL;
	cl_uint			ret_num_devices;
	cl_uint			ret_num_platforms;
	cl_int			ret;

	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

	// создание контекста и очереди команд
	cl_context			context;
	cl_command_queue	command_queue;

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

	// создание объектов памяти для каждого массива
	cl_mem	a_mem_obj;
	cl_mem	b_mem_obj;
	cl_mem	c_mem_obj;

	a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, arr_size * sizeof(cl_int), NULL, &ret);
	b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, arr_size * sizeof(cl_int), NULL, &ret);
	c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, arr_size * sizeof(cl_int), NULL, &ret);

	// запись информации в буферы памяти
	ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, arr_size * sizeof(int), A, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, arr_size * sizeof(int), B, 0, NULL, NULL);

	// создание и компиляция программы
	cl_program	program;
	cl_kernel	kernel;

	program = clCreateProgramWithSource(context, 1, (const char**)&src_str, (const size_t*)&src_size, &ret);	// создать программу из изходного кода (файл kernel.cl)
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);												// билд проги
	kernel = clCreateKernel(program, "addition", &ret);															// создание кернеля

	// установка аргументов кернеля
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);	// массив A
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);	// массив B
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);	// массив C

	// запуск OpenCL
	size_t	NDRange;	// размер вычислительной сетки, общее кол-во элементов, которое нужно посчитать
	size_t	work_size;	// размер word group

	NDRange = arr_size;
	work_size = 64;		// NDRange должен быть кратен размеру work group

	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &NDRange, &work_size, 0, NULL, NULL);	// исполнение кернеля

	// считать результат обратно в хост
	int* C;
	C = (int*)malloc(sizeof(int) * arr_size);

	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, arr_size * sizeof(int), C, 0, NULL, NULL);	// запись ответов в массив C

	printf("DONE!\n");

	if (C != NULL) {
		// вывод результатов в консоль
		for (i = 0; i < arr_size; i++)
		{
			printf("%d + %d = %d\n", A[i], B[i], C[i]);
		}
	}

	// очистка памяти
	ret = clFlush(command_queue);	// очистка очереди комманд
	ret = clFinish(command_queue);	// завершить выполнение всех команд в очереди

	ret = clReleaseKernel(kernel);		// удалить kernel
	ret = clReleaseProgram(program);	// удалить программу OpenCL

	ret = clReleaseMemObject(a_mem_obj);	// очистка буфера для массива A
	ret = clReleaseMemObject(b_mem_obj);	// очистка буфера для массива B
	ret = clReleaseMemObject(c_mem_obj);	// очистка буфера для массива C

	ret = clReleaseCommandQueue(command_queue);		// удалить очередь команд
	ret = clReleaseContext(context);				// удалить контекст OpenCL

	free(A);	// удалить массив A
	free(B);	// удалить массив B
	free(C);	// удалить массив C

	return(0);
}