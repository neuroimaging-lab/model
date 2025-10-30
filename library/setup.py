from setuptools import find_packages, setup

setup(
    name="abnormality_detection_from_neuroimaging",
    version="1.0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "abnormality_detection_from_neuroimaging": ["3DU-Net-model.pt"],
    },
    install_requires=["torch", "nibabel", "numpy", "pathlib"],
)
