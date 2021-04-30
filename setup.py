from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["env", "mp", "pybullet_planning", "residual_learning", "cic", "cpc"],
    package_dir={"": "python"},
)

setup(**d)
