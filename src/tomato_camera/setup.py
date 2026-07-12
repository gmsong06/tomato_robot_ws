from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tomato_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config/stereo"), glob("config/stereo/*.yaml")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ann Song',
    maintainer_email='ann.song@yale.edu',
    description='Camera nodes for tomato robot',
    license='TODO',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "camera_node = tomato_camera.camera_node:main",
            "disparity_viewer_node = tomato_camera.disparity_viewer_node:main",
        ],
    },
)