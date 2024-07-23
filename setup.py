#=================================================================================
#!/usr/bin/python
# encoding: UTF-8
#
# FILE: setup.py
#
#
# DESCRIPTION:
#
# OPTIONS:
# REQUIREMENTS:
# AUTHOR: Khanh Nguyen Gia, khanh@mecanica.upm.es or khanhng1982@yahoo.com
# WEB: http://w3.mecanica.upm.es/~khanh
# VERSION: 1.0.10
# CREATED: 17-10-2017
# LICENSE: MIT
#=================================================================================

"""
A setuptools based setip module
"""

# Using the setuptools to setup package
from setuptools import setup, find_packages

# function that reads the README file
def readme():
    with open('README.md') as f:
        return f.read()

# Defining the setup configurations
setup(
    # name of the package
    name='caldintav',
    # version of the program
    version='3.0',
    # its brief description
    description='CALDINTAV program to calculate the dynamic response of bridges',
    # its long description
    long_description=readme(),
    # the program main home page
    url='http://www.mecanica.upm.es',
    # authors  of program
    author='K. Nguyen; J.M. Goicolea; P. Antolin',
    # autho emails
    author_email = 'khanh@mecanica.upm.es',
    # chooose license
    license='GPL-3.0',
    # include developed packages into the module
    packages=find_packages(),
    # Dependencies of packages. When the program is installed by "pip", this is the
    # specification that is needed to install its dependencies
    install_requires=['numpy>=1.11','sympy>=0.7','joblib>=0.9','scipy>=0.17','matplotlib>=1.5','pyqt5'],
    # Requires a certain Python version
    python_requires='>=3.0',
    # To provide executable scripts, use entry points that provide cross-platform
    # support and allow "pip" to create the appropiate form of executable for the target platform
    entry_points={
        'console_scripts': ['caldintav3=caldintav.runs:run_gui'],
        },
    include_package_data=True,
    zip_safe=False
    )
