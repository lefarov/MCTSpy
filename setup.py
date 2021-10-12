from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mctspy", # Replace with your own username
    version="0.0.1",
    author="Maksym Lefarov",
    author_email="mlefarov@gmail.com",
    description="MCTS implementation in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lefarov/MCTSpy",
    packages=find_packages(exclude=["docs", "tests"]),    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)