from setuptools import find_packages, setup

package_name = 'casper'


setup(
    name=package_name,
    packages=find_packages(
        where='.',
        include=[f'{package_name}*']
    ),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    zip_safe=True,
    maintainer='voshch',
    maintainer_email='dev@voshch.dev',
    description='CASPER++ multi-agent task allocation core: goals, team skills, allocation and schedule',
    license='TODO: License declaration',
)
