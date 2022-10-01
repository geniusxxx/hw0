#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cassert>
#include <cmath>
#include <iostream>


namespace py = pybind11;

float *matric_matmul(const float *X, const float *Y, size_t x_rows, size_t x_cols, size_t y_rows, size_t y_cols)
{ 
	assert(x_cols == y_rows);
	size_t i, j, l;
	float *Z = new float[x_rows * y_cols](); 
	for (i = 0; i < x_rows; i++) {
		for (j = 0; j < y_cols; j++) {
			//float z = 0;
			for (l = 0; l < y_rows; l++) {
				//z += X[i * y_rows + l] * Y[l * y_cols + j];
				Z[i * y_cols + j] += X[i * y_rows + l] * Y[l * y_cols + j];
			} 
			//Z[i * y_cols + j] = z; 
		}
	}
	return Z;
}

/*float *matric_dot_division(const float *X, const float *Y, size_t rows, size_t cols)
{
	size_t i;
	float *Z = new float[rows * cols];
	for (i = 0; i < rows * cols; i++) {
		Z[i] = X[i] / Y[i];
	}
	return Z;
}*/

/*float *matric_exp(const float *X, size_t rows, size_t cols)
{
	size_t i;
	float *Z = new float[rows * cols];
	for (i = 0; i < rows * cols; i++) {
		Z[i] = exp(X[i]);
	}
	return Z;
}*/

float *matric_softmax(const float *X, size_t rows, size_t cols)
{
	size_t i, j;
	float *Z = new float[rows * cols];
	for (i = 0; i < rows; i++) {
		float z = 0.;
		for (j = 0; j < cols; j++) {
			Z[i * cols + j] = exp(X[i * cols + j]);
			z += Z[i * cols + j];
		}
		for (j = 0; j < cols; j++) {
			Z[i * cols + j] /= z;
		}
	}
	return Z;
}

/*float *matric_softmax(float *H, size_t rows, size_t cols)
{
	float *H_exp = matric_exp(H, rows, cols);
	float *H_sum = matric_sum_rows_broadcast(H_exp, rows, cols);
	float *H_times = matric_dot_division(H_exp, H_sum, rows, cols);
	return H_times;
}*/

float *matric_transpose(float *X, size_t rows, size_t cols)
{
	size_t i, j;
	float *X_T = new float[cols * rows];
	for(i = 0; i < rows ; i++) {
		for (j = 0; j < cols; j++) {
			X_T[j * rows + i] = X[i * cols + j];
		}
	}
	return X_T;
}

float *matric_minus(const float *X, const float *Y, size_t rows, size_t cols)
{
	float *M = new float[rows * cols];
	size_t i;
	for (i = 0; i < rows * cols; i++) {
			M[i] = X[i] - Y[i];
	}
	return M;
}

void matric_minus_itself(float *X, const float *Y, size_t rows, size_t cols)
{
	size_t i;
	for (i = 0; i < rows * cols; i++) {
		X[i] -= Y[i];  
	}
}

float *matric_scalar_times(const float *X, size_t rows, size_t cols, const float scalar)
{
	float *S = new float[rows * cols];
	size_t i;
	for (i = 0; i < rows * cols; i++) {
		S[i] = X[i] * scalar;
	}
	return S;
}

float *matric_I(const unsigned char *y,  size_t rows, size_t cols)
{
	size_t i;
	float *I = new float[rows * cols](); 
	for (i = 0; i < rows; i++) {
		I[i * cols + y[i]] = 1.;
	}
	return I;
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
	size_t indices = 0;
	while(indices < m) {
	    size_t batch_size;
	    if (indices + batch <= m) {
	        batch_size = batch;
	    } 
	    else {
	        batch_size = m - indices;
	    }
		//size_t batch_size = min(batch ,m - indices);
		float *X_batch = new float[batch_size * n];
		unsigned char *y_batch = new unsigned char[batch_size];
		memcpy(X_batch, X + indices * n, batch_size * n * sizeof(float));
		memcpy(y_batch, y + indices, batch_size * sizeof(unsigned char));
		float *h = matric_matmul(X_batch, theta, batch_size, n, n, k);
		float *Z = matric_softmax(h, batch_size, k);
		float *I_y = matric_I(y_batch, batch_size, k);
		//float *X_T = matric_transpose(X_batch, batch_size, n);
		//float *Z_minus_I_y = matric_minus(Z, I_y, batch_size, k);
		float *grad = matric_matmul(matric_transpose(X_batch, batch_size, n), matric_minus(Z, I_y, batch_size, k), n, batch_size, batch_size, k);
		float *grad_batch = matric_scalar_times(grad, n, k, lr / batch_size);
		matric_minus_itself(theta, grad_batch, n, k);
		indices += batch_size;  
	}
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
