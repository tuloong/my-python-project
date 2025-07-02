"""
Stock Prediction Project Setup
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements文件
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="stock-prediction-csi300",
    version="1.0.0",
    author="Stock Prediction Team",
    author_email="your-email@example.com",
    description="沪深300股票价格预测机器学习项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-prediction",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/stock-prediction/issues",
        "Documentation": "https://github.com/yourusername/stock-prediction/wiki",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.910",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "streamlit>=1.0.0",
            "dash>=2.0.0",
        ],
        "api": [
            "fastapi>=0.70.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0",
        ],
        "database": [
            "sqlalchemy>=1.4.0",
            "psycopg2-binary>=2.9.0",
            "pymongo>=4.0.0",
            "redis>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "stock-predict=src.scripts.predict:main",
            "stock-train=src.scripts.train:main",
            "stock-data=src.scripts.data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
        "src": ["configs/*.yaml", "configs/*.yml"],
    },
    zip_safe=False,
    keywords=[
        "stock prediction",
        "machine learning",
        "finance",
        "CSI 300",
        "沪深300",
        "algorithmic trading",
        "quantitative finance",
    ],
) 