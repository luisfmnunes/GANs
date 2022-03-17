from setuptools import setup, find_packages

def get_requirements(path : str = "requirements.txt"):
    """
    read requirements.txt file and split the required libraries for setup installation
    """
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines()]

setup(
    name = "GANs",
    version = "0.0.1",
    author="Luís Felipe de Melo Nunes",
    author_email="luisfelipe5417@gmail.com",
    description="A study package implementing GANs",
    packages=find_packages(),
    install_requires=get_requirements()
)