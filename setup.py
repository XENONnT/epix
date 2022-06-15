import setuptools

with open('requirements.txt') as f:
    requires = f.read().splitlines()

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

setuptools.setup(
    name='epix',
    version='0.3.1',
    description='Electron and Photon Instructions generator for XENON',
    author='epix contributors, the XENON collaboration',
    url='https://github.com/XENONnT/epix',
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    setup_requires=['pytest-runner'],
    install_requires=requires,
    tests_require=requires + ['pytest'],
    python_requires=">=3.6",
    extras_require={
        'docs': [
            'sphinx',
            'sphinx_rtd_theme',
            'nbsphinx',
            'recommonmark',
            'graphviz']},
    scripts=['bin/run_epix'],
    packages=setuptools.find_packages(),
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics'],
    zip_safe=False)
