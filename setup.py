from setuptools import setup, find_packages
import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()
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
      install_requires=install_requires
)
