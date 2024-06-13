from setuptools import setup

required_pkgs = ["numpy",
                 "scipy",
                 "typing",
                 "matplotlib",
                 "cmasher",
                 #"h5py",
                 #"pathos",
                 'importlib-metadata; python_version>"3.7"'
                ]

setup(
    name='hkb_utils',
    version='0.1',
    packages=["hkb_utils", ],
    description='project utilities',
    long_description=open('README.md').read(),
    author='Harshvardhan K. Babla',
    author_email='harshbabla@gmail.com',
    url='harshbabla@gmail.com',
    install_requires=required_pkgs,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: >=3.7',
    ],
)