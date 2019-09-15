from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'revtorch', 
  packages = ['revtorch'],
  version = '0.2.0', 
  license='bsd-3-clause',
  description = 'Framework for creating (partially) reversible neural networks with PyTorch',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author = 'Robin Brügger',
  author_email = 'brueggerrobin+revtorch@gmail.com',
  url = 'https://github.com/RobinBruegger/RevTorch',
  download_url = 'https://github.com/RobinBruegger/RevTorch/archive/v0.2.0.tar.gz',
  keywords = ['reversbile neural network'],
  install_requires=[],
  classifiers=[
    'Development Status :: 4 - Beta',
	'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
	'Programming Language :: Python :: 3.7',
	'Programming Language :: Python :: 3.8',
  ],
)