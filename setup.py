from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='lmc_external_kv_service_sm_backend',
    version='0.1.0',
    packages=find_packages(),
    author='LMCache Contributors',
    author_email='tlohit@amzon.com',
    description='External KVServiceSM backend implementation for LMCache',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/lmcache/lmc_external_kv_service_sm_backend',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',

)
