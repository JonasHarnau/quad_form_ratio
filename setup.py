from setuptools import setup

setup(name='quad_form_ratio',
      version='0.1.0',
      description='Ratios of Normal Quadratic Forms',
      url='http://github.com/JonasHarnau/quad_form_ratio',
      author='Jonas Harnau',
      author_email='j.harnau@outlook.com',
      license='MIT',
      packages=['quad_form_ratio'],
      install_requires=['numpy', 'scipy', 'pandas'],
      python_requires='>=3.6',
      include_package_data=True,
      classifiers=['Development Status :: 3 - Alpha'],
      long_description=open('README.md').read(),
      zip_safe=False)