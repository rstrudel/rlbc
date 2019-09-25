import shutil

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))


def read_requirements_file(filename):
    req_file_path = path.join(path.dirname(path.realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='learning2manipulate',
    version='1.0.0',
    description='Learning long-horizon robotic manipulations',
    packages=find_packages(),
    install_requires=read_requirements_file('requirements.txt'))

shutil.copyfile('bc/settings_template.py', 'bc/settings.py')
shutil.copyfile('ppo/settings_template.py', 'ppo/settings.py')
print('In order to make the repo to work, modify bc/settings.py and ppo/settings.py')
