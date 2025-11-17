from setuptools import find_packages, setup

setup(
    name="neuroimaging",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=False,
    install_requires=[
        "torch",
        "nibabel",
        "numpy",
        "huggingface_hub",
    ],
)
