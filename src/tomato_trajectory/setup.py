from setuptools import find_packages, setup

package_name = 'tomato_trajectory'

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
            "record_trajectory = tomato_trajectory.record_trajectory_node:main",
            "replay_trajectory = tomato_trajectory.replay_trajectory_node:main",
        ],
    },
)
