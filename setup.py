import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reward_surfaces",
    version="0.0.1",
    author="Benjamin Black",
    author_email="benblack769@gmail.com",
    description="Reaserch tool for understanding reinforcement learning optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weepingwillowben/reward-surfaces",
    keywords=["Machine Learning", "Job Scheduling"],
    packages=["reward_surfaces"]+["reward_surfaces."+pkg for pkg in setuptools.find_packages("reward_surfaces")],
    install_requires=[
        "plotly",
        "opencv-python",
        "atari-py",
        "stable_baselines3",
        "torch",
        "tqdm",
        "seaborn",
        "h5py",
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
