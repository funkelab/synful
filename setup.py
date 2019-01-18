from distutils.core import setup

setup(
        name='synful',
        version='0.1dev',
        description='Synaptic Partner Detection in 3D Microscopy Volumes.',
        url='https://github.com/funkelab/synful',
        license='MIT',
        packages=[
            'synful',
            'synful.gunpowder',
        ]
)
