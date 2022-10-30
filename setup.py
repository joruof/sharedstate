from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "sharedstate",
        sorted(glob("src/*.cpp")),
        extra_link_args=["-lrt"]
    ),
]

setup(name="sharedstate",
      version="0.0.1",
      description="Simple shared application state",
      author="Jona Ruof",
      author_email="jona.ruof@uni-ulm.de",
      license="MIT",
      zip_safe=False,
      python_requires=">=3.6",
      ext_modules=ext_modules,
      include_package_data=True,
      install_requires=[
            "pybind11",
          ]
      )
