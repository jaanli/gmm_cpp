#include "link_function.hpp"

std::unordered_map<string, shared_ptr<LinkFunction> > lf_map;

LinkFunction*  get_link_function(const string& lf_name) {
  if (lf_map.count("softplus") == 0)
    lf_map["softplus"] = shared_ptr<LinkFunction>( new SoftPlus() );
  if (lf_map.count("id") == 0)
    lf_map["id"] = shared_ptr<LinkFunction>( new IdentityLink() );
  assert(lf_map.count(lf_name) > 0);
  return &*lf_map.at(lf_name);
}
