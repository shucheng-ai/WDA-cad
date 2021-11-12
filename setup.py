#!/usr/bin/env python3
import sys
import os
import subprocess as sp
# monkey-patch for parallel compilation
"""
def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    # parallel code
    N=8 # number of parallel compilations
    import multiprocessing.pool
    def _single_compile(obj):
        try: src, ext = build[obj]
        except KeyError: return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).map(_single_compile,objects))
    return objects
import distutils.ccompiler
distutils.ccompiler.CCompiler.compile=parallelCCompile
"""
from distutils.core import setup, Extension

cvlibs = [a[2:] for a in sp.check_output('pkg-config --libs opencv', shell=True).decode('ascii').strip().split()]

libraries = []
layout = Extension('cad_core',
        language = 'c++',
        extra_compile_args = ['-O3', '-std=c++1y', '-I.', '-g', '-Wno-sign-compare', '-Wno-parentheses'], 
        include_dirs = ['/usr/local/include', 'include'],
        libraries = cvlibs + ['glog'],
        library_dirs = ['/usr/local/lib', 'src'],
        sources = ['cad.cpp', 'geometry.cpp']
        )

setup (name = 'cad_core',
       version = '0.001',
       author = '数程科技',
       author_email = 'future@shucheng.ai',
       license = 'proprietary',
       description = '',
       ext_modules = [layout],
       )

#os.system("./make.sh") # ./make.sh 调用setup.py build和make
