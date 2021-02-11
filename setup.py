from setuptools import find_packages, setup
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='idcempy',  # How you named your package folder (MyLib)
    packages=['idcempy'],  # Chose the same as "name"
    version='0.0.1',  # Start with a small number and increase it
    packages=find_packages(),
    license='mit',  # https://help.github.com/articles/licensing-a-repository
    description='Inflated Discrete Choice Models',  # Give description
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Nguyen Huynh, Sergio Bejar, Vineeta Yadav, Bumba Mukherjee',  # Type in your name
    author_email='nguyenhuynh831@gmail.com',  # Type in your E-Mail
    url='https://github.com/hknd23/idcempy',  # Link to GitHub/website
    keywords=['Inflated', 'Mixture', 'Ordered Probit', 'Multinomial Logit'],  # Keywords
    install_requires=[  # I get to this in a second
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
