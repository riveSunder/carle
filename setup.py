from setuptools import setup

setup(name="carle",\
        py_modules=["carle", "test", "evaluation"],\
        install_requires=["numpy==1.18.4",\
                        "torch==1.5.1",\
                        "scikit-image==0.17.2",\
                        "matplotlib==3.3.3"],\
        version="0.01")




