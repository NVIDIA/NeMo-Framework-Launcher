from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='nemo_megatron_launcher',
    version='0.0.0',
    author='NVIDIA',
    description='NeMo Megatron launcher and tools',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NVIDIA/NeMo-Megatron-Launcher',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'lint': [
            'black==19.10b0',
            'click==8.0.2',
        ],
        'test': [
            'pytest',
            'requests-mock',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
