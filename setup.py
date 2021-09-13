from setuptools import setup, find_namespace_packages

setup(
    name='st3',
    version='1',
    python_requires='>=3.9',
    author='Vol Ki',
    author_email='vol@wilab.org.ua',
    packages=find_namespace_packages(include=['st3.*']),
    long_description="STT for TorchScript",
    install_requires=[
        #"lhotse",
        #"tensorflow-macos",
        "torch",
        "torchaudio",
    ]
)
