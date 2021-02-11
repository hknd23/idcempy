from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='idcempy',
    packages=['idcempy'],
    version='0.0.1',
    packages=find_packages(),
    license='mit',
    description='Inflated Discrete Choice Models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Nguyen Huynh, Sergio Bejar, Vineeta Yadav, Bumba Mukherjee',
    author_email='nguyenhuynh831@gmail.com',
    url='https://github.com/hknd23/idcempy',
    keywords=['Inflated', 'Mixture', 'Ordered Probit', 'Multinomial Logit'],
    install_requires=[
        'scipy',
        'numpy',
        'pandas'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
    ],
)
