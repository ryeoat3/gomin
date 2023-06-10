from setuptools import setup, find_packages

setup(
    name="gaudio_open_vocoder",
    version="0.1.0",
    packages=find_packages(include=["gomin", "gomin.*"]),
    install_requires=[
        "torch==1.13.1",
        "torchaudio==0.13.1",
        "diffusers==0.17.0",
        "librosa>=0.9.2",
        "scipy>=1.10.0",
        "soundfile>=0.11.0",
        "numpy==1.24.3",
        "pyyaml",
        "tqdm",
    ],
)
