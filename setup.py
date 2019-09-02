# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
from setuptools.command.install import install

def _post_install():
    """Post installation nltk corpus downloads."""
    import nltk
    nltk.download("punkt")
    nltk.download("stopwords")

class PostInstall(install):
    """Post-installation for production mode."""
    def run(self):
        install.run(self)
        self.execute(_post_install, [], msg="Running post installation tasks")

here = path.abspath(path.dirname(__file__))

# # Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name='textwrangler',  # Required
    version='0.0.1',  # Required
    description='A simple library for cleaning and pre-processing text.',  # Optional
    url='https://github.com/mattmurray/textwrangler',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),  # Required
    python_requires='>=3.6',
    install_requires=['nltk',
                      'beautifulsoup4',
                      'better_profanity',
                      'contractions',
                      'langdetect',
                      'textblob',
                      'textstat',
                      'textsearch',
                      'inflect',
                      'unidecode',
                      'scikit-learn'
                      ],  # Optional
    setup_requires=['nltk'],
    cmdclass={"install": PostInstall},
    )

