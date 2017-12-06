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

%apply (float** ARGOUTVIEW_ARRAY2, int* DIM1, int* DIM2) {\
(float** train_data, int* train_data_dim1, int* train_data_dim2)};

%apply (float** ARGOUTVIEW_ARRAY2, int* DIM1, int* DIM2) {\
(float** train_labels, int* train_labels_dim1, int* train_labels_dim2)};
                                                          
%apply (int** ARGOUTVIEW_ARRAY3, int* DIM1, int* DIM2, int* DIM3) {\
(int** closest, int* dim1, int* dim2, int* dim3)};

namespace std {
   %template(VectorInt) vector<int>;
   %template(VectorFloat) vector<float>;
};

%include "kmeans2d.h"
