from setuptools import setup

setup(name="carle",\
        packages=["carle", "tests", "evaluation"],\
        install_requires=["numpy==2.4.2",\
                        "torch==2.10.0",\
                        "scikit-image==0.26.0",\
                        "matplotlib==3.10.8"],\
        version="0.01")




