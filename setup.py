import os

from setuptools import find_packages, setup

with open(os.path.join("esbcpo", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()


long_description = """
todo
"""


setup(
    name="esbcpo",
    packages=[package for package in find_packages() if package.startswith("esbcpo")],
    package_data={"esbcpo": ["py.typed", "version.txt"]},
    install_requires=[
        
    ],
    description="Implement of ESB-CPO",
    author="Tsinghua Space Robot Learning Group",
    url="https://github.com/Tsinghua-Space-Robot-Learning-Group/ESB-CPO",
    author_email="xuht21@mails.tsinghua.edu.cn",
    keywords="Extra Safety Budget for Safe RL", 
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)