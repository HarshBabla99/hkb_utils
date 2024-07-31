from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'HKB Utils'
LONG_DESCRIPTION = 'Utilities for quantum information theory by HKB'

required_pkgs = ["numpy",
                 "scipy",
                 "typing",
                 "matplotlib",
                 "cmasher",
                 "qutip",
                 "dynamiqs",
                 "imageio",
                 #"h5py",
                 #"pathos",
                 'importlib-metadata; python_version>"3.7"'
                ]

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="hkb_utils", 
        version=VERSION,
        author="Harshvardhan K. Babla",
        author_email="<harsh.babla@yale.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=required_pkgs,        
        keywords=['python', 'hkb_utils'],
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Programming Language :: Python :: >=3.7',
        ],
)