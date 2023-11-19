#!/bin/bash
#!/usr/bin/env python3

python setup.py bdist_wheel
cp dist/restaurant_demo-0.1-py3-none-any.whl flask_dockerfile/python_wheels