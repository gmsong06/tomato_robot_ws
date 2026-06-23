from setuptools import find_packages, setup
import os
from glob import glob

package_name = "tomato_bringup"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ann Song",
    maintainer_email="ann.song@yale.edu",
    description="Bringup launch files for tomato robot",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)