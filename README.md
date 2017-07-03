# gmm_cpp

Gaussian mixture model implementation in C++ with black box variational inference and control variates

This is heavily based on the deep exponential families repo, a beautiful piece of software: https://github.com/blei-lab/deep-exponential-families

## Requirements
On a mac:
```
brew install gcc --without-multilib
brew install gsl
brew install homebrew/science/armadillo
brew install boost
```

On ubuntu: see my docker file [here](https://github.com/altosaar/dotfiles/blob/master/.docker/Dockerfile).

# Running

```
./waf configure
./waf build
./build/my_main
```

This runs black box variational inference to fit a gaussian mixture model to toy data.
