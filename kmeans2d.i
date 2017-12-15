%module kmeans2d

 %{
#define SWIG_FILE_WITH_INIT
#include "kmeans2d.h"
%}

%include "std_vector.i"
%include "std_string.i"
%include "typemaps.i"
%include "numpy.i"

%init %{
    import_array();
%}

%apply int *OUTPUT { int* width };
%apply int *OUTPUT { int* height };

%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {\
(float** train_data, int* train_data_dim1, int* train_data_dim2)};

%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {\
(float** train_labels, int* train_labels_dim1, int* train_labels_dim2)};
                                                          
%apply (int** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {\
(int** closest, int* dim1, int* dim2, int* dim3)};

%apply (int** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {\
(int** average, int* average_dim1, int* average_dim2, int* average_dim3)};

%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {\
(float* closest, int closest_dim1, int closest_dim2, int closest_dim3)};

%apply (float* INPLACE_ARRAY2, int DIM1,int DIM2) {\
(float* train_data, int train_data_dim1, int train_data_dim2)};

%apply (float* INPLACE_ARRAY2, int DIM1,int DIM2) {\
(float* target_data, int target_data_dim1, int target_data_dim2)};

%apply (float* ARGOUTVIEWM_ARRAY2, int DIM1,int DIM2) {\
(float** test, int* test_dim1, int* test_dim2)};

%apply (float* ARGOUTVIEWM_ARRAY2, int DIM1,int DIM2) {\
(float** target, int* target_dim1, int* target_dim2)};

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(float* test, int test_dim1, int test_dim2)};

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(float* target, int target_dim1, int target_dim2)};

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(float* predictions, int predictions_dim1, int predictions_dim2)};

%apply (float* INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4) {\
(float* predicted_images, int predicted_images_dim1, int predicted_images_dim2,\
int predicted_images_dim3, int predicted_images_dim4)};

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(float* errors, int errors_dim1, int errors_dim2)};

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(float* totals, int totals_dim1, int totals_dim2)};

/*void compute_errors(std::vector<int>& indices, float* train, int train_dim1,*/
                    /*int train_dim2, float* target, int target_dim1,*/
                    /*int target_dim2, float* predictions, int predictions_dim1,*/
                    /*int predictions_dim2, float* predicted_images,*/
                    /*int predicted_images_dim1, int predicted_images_dim2,*/
                    /*int predicted_images_dim3, int predicted_images_dim4,*/
                    /*float* errors, int errors_dim1, int errors_dim2);*/
namespace std {
   %template(VectorInt) vector<int>;
   %template(VectorFloat) vector<float>;
};

%include "kmeans2d.h"
