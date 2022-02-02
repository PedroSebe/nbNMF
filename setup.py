from setuptools import setup

setup(name='nbNMF',
      version='0.0.1',
      description='nbNMF is an easy-to-use Python library for robust dimensionality reduction with count data',
      url='https://github.com/PedroSebe/nbNMF',
      author='Pedro Sebe',
      license='MIT',
      packages=['nbNMF'],
      install_requires=["numpy","scipy","sklearn"],
      zip_safe=False)