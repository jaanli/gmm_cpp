#pragma once

#include "utils.hpp"
#include <cassert>
#include <cmath>
#include <unordered_map>


using namespace std;

struct LinkFunction {
  virtual double f(double v) = 0;
  virtual double g(double v) = 0;
  virtual double f_inv(double v) = 0;
};

struct SoftPlus : public LinkFunction {
  virtual double f(double x) {
    if (x < -5)
      return exp(x);
    else if (x > 10)
      return x;
    else
      return log(1+exp(x));
  }
  virtual double g(double x) {
    if (x > 10)
      return 1;
    else
      return exp(x) / (1 + exp(x));
  }
  virtual double f_inv(double y) {
    assert(y > 0);
    if (y > 10)
      return y;
    else
      return log(exp(y) - 1);
  }
};

struct IdentityLink : public LinkFunction {
  virtual double f(double x) {
    return x;
  }
  virtual double g(double x) {
    return 1;
  }
  virtual double f_inv(double y) {
    return y;
  }
};

LinkFunction*  get_link_function(const string& lf_name);
