from setuptools import setup, find_packages


setup(
    name='wsisurv',
    version='0.1.0',
    description='WSI_SURV',
    url='https://github.com/ezgiogulmus/WSI_Surv',
    author='FEO',
    author_email='',
    license='GPLv3',
    packages=find_packages(exclude=['assets', 'datasets_csv', "splits"]),
    install_requires=[
        "torch>=2.3.0",
        "numpy==1.23.4", 
        "pandas==1.4.3",
        "h5py",
        "nystrom_attention",
        "scikit-learn", 
        "scikit-survival",
        "tensorboardx",
        "topk @ git+https://github.com/oval-group/smooth-topk.git",
        "future"
    ],

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: GPLv3",
    ]
)