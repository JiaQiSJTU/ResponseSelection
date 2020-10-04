#!/usr/bin/env python3


from setuptools import setup, find_packages
import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python >=3.6 is required for ParlAI.')

readme = ""
license = ""
with open('requirements.txt') as f:
    reqs = f.read()


if __name__ == '__main__':
    setup(
        name='parlai',
        version='0.1.0',
        description='Unified API for accessing dialog datasets.',
        long_description=readme,
        url='http://parl.ai/',
        license=license,
        python_requires='>=3.6',
        packages=find_packages(
            exclude=('data', 'docs', 'downloads', 'examples', 'logs', 'tests')
        ),
        install_requires=reqs.strip().split('\n'),
        include_package_data=True,
        test_suite='tests.suites.unittests',
    )
