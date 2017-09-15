%module kmeans2d

 %{
#include "kmeans2d.h"
 %}

%include "std_vector.i"

namespace std {
   %template(VectorInt) vector<int>;
   %template(VectorFloat) vector<float>;
};

%include "kmeans2d.h"
