def options(opt):
  opt.load('compiler_cxx')
  opt.add_option('--mode', action='store', default='debug', help='Compile mode: release or debug')
  opt.add_option('--exe', action='store_true', default=False, help='Execute program after build')

def configure(conf):
  print("configure!")

  # Set the compiler to waf
  # on mac, install with `brew install gcc --without-multilib`
  conf.env.CXX = 'g++-7'
  conf.env.CC = 'gcc-7'
  # sudo apt-get install libgsl2 libgsl-dev libarmadillo6 libarmadillo-dev libboost-all-dev
  # or homebrew
  conf.env.LIBPATH_MYLIB = ['/usr/lib', '/usr/lib64']

  if conf.options.mode == 'release':
    cxx_flags = ['-O3', '-fpermissive', '-std=c++11', '-march=native', '-DNDEBUG', '-fopenmp', '-DBOOST_LOG_DYN_LINK']
  else:
    print('debug mode')
    cxx_flags = ['-O0', '-g', '-ggdb', '-std=c++11', '-fopenmp', '-DBOOST_LOG_DYN_LINK']


  conf.env.append_value('CXXFLAGS', cxx_flags)

  conf.load('compiler_cxx')
  conf.check(compiler='cxx',lib=['m', 'gsl', 'gslcblas'], uselib_store='GSL')
  conf.check(compiler='cxx',lib='pthread', uselib_store='PTHREAD')
  conf.check(compiler='cxx',lib='gomp', uselib_store='OPENMP')
  conf.check(compiler='cxx',lib='armadillo', uselib_store='ARMADILLO')
  conf.check(compiler='cxx',lib='boost_program_options', uselib_store='PROGRAM_OPTIONS')
  conf.check(compiler='cxx',lib='boost_iostreams', uselib_store='IOSTREAMS')
  conf.check(compiler='cxx',lib='boost_serialization', uselib_store='SERIALIZATION')
  conf.check(compiler='cxx',lib='boost_filesystem', uselib_store='FILESYSTEM')
  conf.check(compiler='cxx',lib='boost_system', uselib_store='SYSTEM')
  conf.check(compiler='cxx',lib='boost_log', uselib_store='LOG')
  conf.check(compiler='cxx',lib='boost_random', uselib_store='RANDOM')

def post(ctx):
  if ctx.options.exe:
    ctx.exec_command('./build/my_main')

def build(bld):
  src = [
        # 'dirichlet_main.cpp',
  	 'gaussian_mixture_main.cpp',
	 'data.cpp',
	 'optimizer.cpp',
	 'bbvi.cpp',
	 'link_function.cpp',
	 'serialization.cpp',
	 'variational_inference.cpp']

  # lib = ['PTHREAD', 'ARMADILLO', 'PROGRAM_OPTIONS', 'IOSTREAMS', 'SERIALIZATION', 'FILESYSTEM', 'SYSTEM', 'OPENMP', 'GSL', 'LOG', 'RANDOM']
  lib = ['ARMADILLO', 'GSL', 'OPENMP', 'SERIALIZATION', 'PROGRAM_OPTIONS', 'PTHREAD']
  bld.program(source=src, use=lib, target='my_main')
  bld.add_post_fun(post)
