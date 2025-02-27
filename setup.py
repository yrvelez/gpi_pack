from setuptools import setup, find_packages

setup(
    name='gpi_pack',
    version='0.1.0',
    description='A package for Generative-AI powered Inference (GPI)',
    author='Kentaro Nakamura',
    author_email='knakamura@g.harvard.edu',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'torch',
        'scikit-learn',
        'pandas',
        'transformers',
        'tqdm',
        'patsy',
        'accelerate',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)