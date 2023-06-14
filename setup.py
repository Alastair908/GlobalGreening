from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='green',
      version="0.0.1",
      description="Global Greening Model",
      license="CC0 1.0 Universal",
      author="Alastair Horne & Bootcamp team",
      # author_email="contact@lewagon.org",
      # url="https://github.com/Alastair908/GlobalGreening/",
      install_requires=requirements,
      packages=find_packages(),
      #test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)