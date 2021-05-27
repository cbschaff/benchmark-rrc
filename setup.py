from setuptools import setup, find_packages
import os

package_name = "rrc"


def find_package_data(base_dir, data_dir):
    """Get list of all files in base_dir/data_dir, relative to base_dir."""
    paths = []
    for (path, directories, filenames) in os.walk(
        os.path.join(base_dir, data_dir)
    ):
        for filename in filenames:
            paths.append(
                os.path.relpath(os.path.join(path, filename), base_dir)
            )
    return paths


setup(
    name=package_name,
    version="1.1.0",
    packages=find_packages("python"),
    package_dir={"": "python"},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    scripts=[
        "scripts/run_episode.py",
        "scripts/run_local_episode.py",
    ],
    include_package_data=True,
    package_data={
        "residual_learning": [
            "*.pt",
            "models/cic_lvl3/logs/*.gin",
            "models/cic_lvl4/logs/*.gin",
            "models/cpc_lvl3/logs/*.gin",
            "models/cpc_lvl4/logs/*.gin",
            "models/mp_lvl3/logs/*.gin",
            "models/mp_lvl4/logs/*.gin",
        ],
        "cic": ["trifinger_mod.urdf"],
    },
)
