#include "gaussian_mixture.hpp"
#include "utils.hpp"
#include "variational_inference.hpp"

int main() {
  pt::ptree options;
  pt::ini_parser::read_ini("options.ini", options);
  PGaussianMixture p_gaussian_mixture(options);
  QGaussianMixture q_gaussian_mixture(options);
  VariationalInference vi(options, &p_gaussian_mixture, &q_gaussian_mixture);
  vi.train();
  return 0;
}
