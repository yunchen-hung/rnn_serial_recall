from setuptools import setup, find_packages

setup(
    name='nmt_cmr_parallels',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'nltk>=3.6.0',
        'gensim>=4.0.0',
        'tensorboard',
        'pandas',
        'gymnasium==0.29.1'
    ],
    author='Nik Salvatore',
    author_email='nds113@scarletmail.rutgers.edu',
    description='Seq2seq model training for free recall tasks on the PEERS dataset.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nds113/nmt_cmr_parallels',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
