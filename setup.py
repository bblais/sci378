import sci378

from distutils.core import setup
from distutils.extension import Extension

setup(
  name = 'sci378',
  version=sci378.__version__,
  description="sci378",
  author="Brian Blais",
  packages=['sci378'],
  install_requires=[
          'matplotlib',
          'pandas',
      ],
)

