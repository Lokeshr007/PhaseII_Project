from setuptools import setup, find_packages

setup(
    name="deepdefend",
    version="1.0.0",
    description="Production Network Intrusion Detection for Imbalanced Networks",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.11.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "deepdefend-train=scripts.train:main",
            "deepdefend-detect=scripts.run_engine:main",
            "deepdefend-evaluate=scripts.evaluate:main",
        ]
    },
    python_requires=">=3.10",
)