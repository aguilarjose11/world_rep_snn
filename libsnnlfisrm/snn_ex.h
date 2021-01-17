#include <exception>

#pragma once

/**
 * Exceptions
*/
class neuronexception: public std::exception
{
  virtual const char* what() const throw();
};

class InputLayerException: public std::exception
{
  virtual const char* what() const throw();
};

class euclideanexception: public std::exception
{
  virtual const char* what() const throw();
};