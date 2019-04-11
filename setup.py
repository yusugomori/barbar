from setuptools import setup
from setuptools import find_packages

setup(
    name='barbar',
    version='0.2.0',
    description='Progress bar for deep learning training iterations',
    author='Yusuke Sugomori',
    author_email='me@yusugomori.com',
    url='https://github.com/yusugomori/barbar',
    download_url='',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='deep learning pytorch tensorflow keras',
    packages=find_packages()
)
