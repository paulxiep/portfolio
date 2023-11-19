from setuptools import setup, find_packages

setup(name='restaurant_demo',
      version="0.1",
      description='',
      long_description='',
      author='Paul',
      author_email='paulxiep@outlook.com',
      url='',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      package_data={'': ['*']},
      py_modules=['restaurant_models', 'synthesize_restaurant_data'],
      install_requires=[
          "catboost==1.2.2",
          "xgboost==2.0.2",
          "scikit-learn==1.3.0"
      ],
      license='Private',
      zip_safe=False,
      keywords='',
      classifiers=[''])
