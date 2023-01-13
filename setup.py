from setuptools import setup, find_packages

setup(
    name='scdfdistributedutils',
    version='2.0',
    packages=find_packages(),
    install_requires=[
        'mlflow==2.0.1',
        'pandas',
        'ray @ https://files.pythonhosted.org/packages/ab/88/dc597fdadd74b1a2fe7d4e4fb56f44a2b444a01b76ec35f31c7eee28f2d0/ray-2.1.0-cp310-cp310-manylinux2014_x86_64.whl',
        # 'ray @ https://files.pythonhosted.org/packages/49/a7/b65ae7d68b4e6f1b5dc52d5ecfe1fcfd437e7377db9fd06f595228be1fa4/ray-2.0.1-cp310-cp310-macosx_10_15_universal2.whl',
        'wheel'
    ]
)
