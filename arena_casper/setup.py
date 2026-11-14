from setuptools import find_packages, setup

package_name = 'arena_casper'


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
    description='CASPER++ multi-agent task allocation as an Arena task layer',
    license='TODO: License declaration',
    entry_points={
        'arena_runtime.task_layer': [
            'casper = arena_casper.layer',
        ],
    }
)
