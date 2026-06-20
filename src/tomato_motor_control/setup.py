from setuptools import setup

package_name = "tomato_motor_control"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ann Song",
    maintainer_email="ann.song@yale.edu",
    description="Motor control package",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "motor_node = tomato_motor_control.motor_node:main",
        ],
    },
)