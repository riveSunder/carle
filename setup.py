from setuptools import setup

setup(name="carle",\
        packages=["carle", "tests", "evaluation"],\
        install_requires=["numpy==1.18.4",\
                        "torch==2.2.0",\
                        "scikit-image==0.17.2",\
                        "matplotlib==3.3.3"],\
        version="0.01")




