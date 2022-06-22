from setuptools import setup

setup(name="carle",\
        packages=["carle", "tests", "evaluation"],\
        install_requires=["numpy==1.22.0",\
                        "torch==1.5.1",\
                        "scikit-image==0.17.2",\
                        "matplotlib==3.3.3"],\
        version="0.01")




