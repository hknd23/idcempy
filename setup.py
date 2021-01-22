from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ziopcpy',  # How you named your package folder (MyLib)
    packages=['ziopcpy'],  # Chose the same as "name"
    version='0.1.2',  # Start with a small number and increase it
    license='mit',  # https://help.github.com/articles/licensing-a-repository
    description='Zero-inflated Ordered Probit Models',  # Give description
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Nguyen Huynh, Sergio Bejar, Nicolas Schmidt, Vineeta Yadav, Bumba Mukherjee',  # Type in your name
    author_email='nkh8@psu.edu',  # Type in your E-Mail
    url='https://github.com/hknd23/ziopcpy',  # Link to GitHub/website
    download_url='https://github.com/hknd23/ziopcpy/archive/v0.1.2.tar.gz',
    keywords=['Zero-Inflated', 'Mixture', 'Ordered'],  # Keywords
    install_requires=[  # I get to this in a second
        'scipy',
        'numpy',
        'pandas'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
    ],
)
