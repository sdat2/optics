from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="Hatan",
    author_email="author@example.com",
    description="optical analysis of eye",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
