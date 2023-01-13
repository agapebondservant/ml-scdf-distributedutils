from setuptools import setup, find_packages

setup(
    name='scdfdistributedutils',
    version='2.0',
    packages=find_packages(),
    install_requires=[
        'mlflow==2.0.1',
        'pandas',
        'ray @ https://files.pythonhosted.org/packages/ab/88/dc597fdadd74b1a2fe7d4e4fb56f44a2b444a01b76ec35f31c7eee28f2d0/ray-2.1.0-cp310-cp310-manylinux2014_x86_64.whl',
        'wheel'
    ]
)
