from setuptools import setup, find_packages

# Override standard setuptools commands. 
# Enforce the order of dependency installation.
#-------------------------------------------------
import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
ordered_install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        ordered_install_requires = f.read().splitlines()

from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

def requires( packages ): 
    from os import system
    from sys import executable as PYTHON_PATH
    from pkg_resources import require
    require( "pip" )
    CMD_TMPLT = '"' + PYTHON_PATH + '" -m pip install %s'
    for pkg in packages: system( CMD_TMPLT % (pkg,) )       

class OrderedInstall( install ):
    def run( self ):
        requires( ordered_install_requires )
        install.run( self )        

class OrderedDevelop( develop ):
    def run( self ):
        requires( ordered_install_requires )
        develop.run( self )        

class OrderedEggInfo( egg_info ):
    def run( self ):
        requires( ordered_install_requires )
        egg_info.run( self )        

CMD_CLASSES = { 
     "install" : OrderedInstall
   , "develop" : OrderedDevelop
   , "egg_info": OrderedEggInfo 
}
#-------------------------------------------------

setup(name='hcl',
      version='1.0dev',
      description='DynAMO for HCL',
      long_description='Dynamic Anticipatory Adaptive Mesh Optimization (DynAMO) for general hyperbolic conservation laws (HCL)',
      author='LLNL',
      classifier=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10"
      ],
      python_requires=">=3.10, <4",
      project_urls={
         "Source": "https://github.com/dohyun-cse/drl4amr/"
      },
      cmdclass=CMD_CLASSES
)
