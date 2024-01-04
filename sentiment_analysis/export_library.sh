#!/bin/bash

python setup.py bdist_wheel
mkdir -p dockerfile/python_wheels
cp dist/paul_sentiment_analysis-0.1-py3-none-any.whl dockerfile/python_wheels
