from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mcts-py", # Replace with your own username
    version="0.0.1",
    author="Maksym Lefaorv",
    author_email="mlefarov@gmail.com",
    description="MCTS implementaion in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lefarov/MCTSpy",
    packages=find_packages(),    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    # install_requires=[
    #     "numpy >= 1.11.1", 
    #     "matplotlib >= 1.5.1".
    # ],
)