from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="turboquantdc",
    version="0.1.0",
    author="TurboQuantDC Contributors",
    description="TurboQuant: 3-bit KV cache compression for LLMs with <0.5% attention quality loss",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/turboquantdc/turboquantdc",
    packages=find_packages(exclude=["tests*", "benchmarks*", "reference*", "docs*", "warroom*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "benchmark": [
            "transformers>=4.40.0",
            "accelerate>=0.25.0",
            "bitsandbytes>=0.43.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    keywords=[
        "llm", "kv-cache", "quantization", "compression",
        "transformer", "attention", "cuda", "pytorch",
    ],
)
