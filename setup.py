from setuptools import setup, find_packages

setup(
    name="textlens",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "pillow>=9.0.0",
        "pypdfium2>=4.20.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
        "requests>=2.28.0",
        "pydantic>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "textlens=textlens.cli:main",
        ],
    },
)
