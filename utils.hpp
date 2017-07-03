// utility functions, included everywhere
#pragma once
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/range/combine.hpp>
#include <iostream>
#include <map>

#include "serialization.hpp"
#include <armadillo>

using namespace std;
typedef vector<arma::uword> ExampleIds;
typedef vector<shared_ptr<arma::rowvec>> VecOfRow;
typedef vector<std::shared_ptr<arma::mat>> VecOfMat;
typedef vector<std::shared_ptr<arma::cube>> VecOfCube;
typedef map<string, VecOfMat> MapVecOfMat;
typedef map<string, std::shared_ptr<arma::mat>> MapOfMat;

namespace pt = boost::property_tree;
