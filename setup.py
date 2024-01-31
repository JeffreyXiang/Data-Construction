import setuptools

setuptools.setup(
    name="data_construction",
    version="0.0.1",
    author="Jianfeng Xiang",
    author_email="belljig@outlook.com",
    description="Utilities for constructing data.",
    long_description="Utilities for constructing data.",
    long_description_content_type="text/markdown",
    url="https://github.com/JeffreyXiang/Data-Construction",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "utils3d",
    ]
)

