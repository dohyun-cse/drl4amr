from setuptools import setup, find_packages
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
      package_dir={"": "hcl"},
      packages = find_packages(where="hcl"),
      python_requires=">=3.10, <4",
      project_urls={
         "Source": "https://github.com/dohyun-cse/drl4amr/"
      }
)
