from setuptools import find_packages, setup
from typing import List

def get_requirements(filepath: str)->List(str):
    requirements = []
    try:
        with open(filepath, 'r') as file:
            for line in file:
                if line.strip() and '-e .' not in line:
                    requirements.append(line.strip())
        return requirements
    except FileNotFoundError as e:
        print(f'Requirements file not found in {filepath}')

setup(
    name='DU Stats Machine Learning Project',
    version='0.0.1',
    author='Bijon Lahiri',
    author_email='bijonlahiri@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)