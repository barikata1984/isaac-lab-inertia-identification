from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'trajectories'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    package_dir={package_name: 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'numpy', 'matplotlib', 'tyro'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='franche1984@gmail.com',
    description='Unified trajectory generation and playback package for Isaac Sim',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_generator = trajectories.generator_cli:entry_point',
            'trajectory_follower = trajectories.follower_node:main',
        ],
    },
)
