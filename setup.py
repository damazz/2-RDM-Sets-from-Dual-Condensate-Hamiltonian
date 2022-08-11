'''
import setuptools

setuptools.setup(
    name="DC",
    version="0.0.1",
    Author="Sam Warren",
    install_requires=["sparse", "matplotlib"],
)
'''
import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'DC'
AUTHOR = 'Sam Warren'
#AUTHOR_EMAIL = 'samwarren@uchicago.edu.com'
#URL = 'https://github.com/you/your_package'

#LICENSE = 'Apache License 2.0'
#DESCRIPTION = 'Describe your package in one sentence'
#LONG_DESCRIPTION = (HERE / "README.md").read_text()
#LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'sparse',
      'matplotlib'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
#      description=DESCRIPTION,
#      long_description=LONG_DESCRIPTION,
#      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
#      license=LICENSE,
#      author_email=AUTHOR_EMAIL,
#      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
