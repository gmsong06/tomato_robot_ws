from setuptools import find_packages, setup

package_name = 'tomato_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ann Song',
    maintainer_email='ann.song@yale.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "tomato_detection_node = tomato_perception.tomato_detection_node:main",
            "tomato_ripeness_node = tomato_perception.tomato_ripeness_node:main",
            "tomato_reactive_controller_node = tomato_perception.tomato_reactive_controller_node:main",
            "depth_probe_node = tomato_perception.depth_probe_node:main",
        ],
    },
)
