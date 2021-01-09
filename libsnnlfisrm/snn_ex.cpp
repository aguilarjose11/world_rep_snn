#include "snn_ex.h"

// Exceptions
const char* neuronexception::what() const throw()
{
  return "Invalid neuron initial values";
}



const char* InputLayerException::what() const throw()
{
return "Invalid inputs for input layer";
}



const char* euclideanexception::what() const throw()
{
return "Invalid euclidean inputs";
}
