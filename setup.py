from setuptools import setup, find_packages


setup(name='DJAA',
      version='0.0.1',
      description='',
      author='Hao Chen',
      author_email='hchen@pku.edu.cn',
      url='https://github.com/chenhao2345/DJAA',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'tqdm', 'imageio'],
      packages=find_packages(),
      keywords=[
          'Contrastive Learning',
          'Person Re-identification'
      ])

