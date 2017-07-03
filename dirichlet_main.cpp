#include "dirichlet.hpp"
#include "utils.hpp"
#include "variational_inference.hpp"

int main() {
  pt::ptree options;
  pt::ini_parser::read_ini("options.ini", options);
  PDirichlet p_dirichlet(options);
  QDirichlet q_dirichlet(options);
  VariationalInference vi(options, &p_dirichlet, &q_dirichlet);
  vi.train();
  cout << "After training, alpha: \n" << q_dirichlet.alpha() << endl;
  return 0;
}
