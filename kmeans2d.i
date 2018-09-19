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
%apply double* OUTPUT {double* accuracy};

%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {\
(double** train_data, int* train_data_dim1, int* train_data_dim2)};

%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {\
(double** train_labels, int* train_labels_dim1, int* train_labels_dim2)};
                                                          
%apply (int** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {\
(int** closest, int* dim1, int* dim2, int* dim3)};

%apply (int** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {\
(int** average, int* average_dim1, int* average_dim2, int* average_dim3)};

%apply (double** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {\
(double** average, int* average_dim1, int* average_dim2, int* average_dim3)};

%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {\
(int* closest, int closest_dim1, int closest_dim2, int closest_dim3)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(double* train_data, int train_data_dim1, int train_data_dim2)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(double* train_labels, int train_labels_dim1, int train_labels_dim2)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(double* target_data, int target_data_dim1, int target_data_dim2)};

%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {\
(double** test, int* test_dim1, int* test_dim2)};

%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {\
(double** target, int* target_dim1, int* target_dim2)};

%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {\
(double** predictions, int* predictions_dim1, int* predictions_dim2)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(double* test, int test_dim1, int test_dim2)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(double* target, int target_dim1, int target_dim2)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(double* train, int train_dim1, int train_dim2)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(double* predictions, int predictions_dim1, int predictions_dim2)};

%apply (double* INPLACE_ARRAY4, int DIM1, int DIM2, int DIM3, int DIM4) {\
(double* predicted_images, int predicted_images_dim1, int predicted_images_dim2,\
int predicted_images_dim3, int predicted_images_dim4)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(double* errors, int errors_dim1, int errors_dim2)};

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {\
(double* totals, int totals_dim1, int totals_dim2)};

%apply (double* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {\
(double* assign, int assign_dim1, int assign_dim2, int assign_dim3)};
  
%apply (uint8_t* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {\
(uint8_t* average, int average_dim1, int average_dim2, int average_dim3)};

%apply (double* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {\
(double* img, int img_dim1, int img_dim2, int img_dim3)};

%apply (int* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {\
(int* assignment_data, int assignment_data_dim1, int assignment_data_dim2,\
 int assignment_data_dim3)};

%apply (double* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {\
(double* average_image, int average_image_dim1, int average_image_dim2,\
 int average_image_dim3)};

%apply (double* INPLACE_ARRAY3, int DIM1, int DIM2, int DIM3) {\
(double* image_out, int image_out_dim1, int image_out_dim2, int image_out_dim3)};

%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {\
(double** test_data, int* test_data_dim1, int* test_data_dim2)};

%apply (int** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {\
(int** ensemble_data, int* ensemble_data_dim1, int* ensemble_data_dim2)};

namespace std {
   %template(VectorInt) vector<int>;
   %template(VectorFloat) vector<double>;
};

%include "kmeans2d.h"
