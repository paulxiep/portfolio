from setuptools import setup, find_packages

setup(name='paul_sentiment_analysis',
      version="0.1",
      description='',
      long_description='',
      author='Paul',
      author_email='paulxiep@outlook.com',
      url='',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      package_data={'':['*']},
      py_modules=['paul_sentiment_analysis'],
      install_requires=[
            "pythainlp==4.0.2",
            "scikit-learn==1.3.2",
            "tensorflow==2.10.1",
            "tensorflow-addons==0.21.0",
            "protobuf==3.19.6"
      ],
      license='Private',
      zip_safe=False,
      keywords='',
      classifiers=[''])
