from setuptools import setup, find_packages
setup(
  name = 'vaderSentiment',
  #packages = ['vaderSentiment'], # this must be the same as the name above
  packages = find_packages(exclude=['tests*']), # a better way to do it than the line above -- this way no typo/transpo errors
  include_package_data=True,
  version = '0.5',
  description = 'VADER Sentiment Analysis. VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, and works well on texts from other domains.',
  author = 'C.J. Hutto',
  author_email = 'cjhutto [at] gatech [dot] edu',
  license = 'MIT License: http://opensource.org/licenses/MIT',
  url = 'https://github.com/cjhutto/vaderSentiment', # use the URL to the github repo
  download_url = 'https://github.com/cjhutto/vaderSentiment/archive/master.zip', 
  keywords = ['vader', 'sentiment', 'analysis', 'opinion', 'mining', 'nlp', 'text', 'data', 
              'text analysis', 'opinion analysis', 'sentiment analysis', 'text mining', 'twitter sentiment',
              'opinion mining', 'social media', 'twitter', 'social', 'media'], # arbitrary keywords
  classifiers = ['Development Status :: 4 - Beta', 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License', 'Natural Language :: English',
                 'Programming Language :: Python :: 2.7', 'Topic :: Scientific/Engineering :: Artificial Intelligence',
                 'Topic :: Scientific/Engineering :: Information Analysis', 'Topic :: Text Processing :: Linguistic',
                 'Topic :: Text Processing :: General'],
)