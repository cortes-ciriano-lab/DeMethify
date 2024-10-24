from setuptools import setup, find_packages

setup(
    name='DeMethify',                      
    version='0.1.0',                          
    description='DeMethify is a partial-reference based methylation deconvolution algorithm that uses a weighted constrained version of an iteratively optimized negative matrix factorization algorithm.',
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',
    author='Marwane Bourdim',
    author_email='marwane.bourdim@icr.ac.uk',
    url='https://github.com/cortes-ciriano-lab/DeMethify',  
    license='MIT',                        
    packages=find_packages(),              
    install_requires=[                         
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scikit-learn==1.2.2',
        'numba==0.59.0',
        'scipy==1.11.4',
        'colorcet',
        'seaborn',
        'matplotlib',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'demethify=demethify.demethify:main',
        ],
    },
    python_requires='>=3.6',                  
    classifiers=[                              
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
