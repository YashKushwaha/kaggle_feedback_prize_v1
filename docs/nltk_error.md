Following error happens when sent_tokenize is used without having punkt_tab
```
Traceback (most recent call last):
  File "/mnt/e/PROJECTS/kaggle_feedback_prize_v1/scripts/create_dataset.py", line 54, in <module>
    sentences = sent_tokenize(full_text)
  File "/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/lib/python3.10/site-packages/nltk/tokenize/__init__.py", line 119, in sent_tokenize
    tokenizer = _get_punkt_tokenizer(language)
  File "/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/lib/python3.10/site-packages/nltk/tokenize/__init__.py", line 105, in _get_punkt_tokenizer
    return PunktTokenizer(language)
  File "/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/lib/python3.10/site-packages/nltk/tokenize/punkt.py", line 1744, in __init__
    self.load_lang(lang)
  File "/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/lib/python3.10/site-packages/nltk/tokenize/punkt.py", line 1749, in load_lang
    lang_dir = find(f"tokenizers/punkt_tab/{lang}/")
  File "/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/lib/python3.10/site-packages/nltk/data.py", line 579, in find
    raise LookupError(resource_not_found)
LookupError:
**********************************************************************
  Resource punkt_tab not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('punkt_tab')
  >>> nltk.download('punkt_tab')

  For more information see: https://www.nltk.org/data.html

  For more information see: https://www.nltk.org/data.html

  Attempted to load tokenizers/punkt_tab/english/

  Attempted to load tokenizers/punkt_tab/english/


  Searched in:
  Searched in:
    - '/root/nltk_data'
    - '/root/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/share/nltk_data'
    - '/root/.cache/pypoetry/virtualenvs/kaggle-feedback-prize-v1-k7f9ZZOq-py3.10/lib/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
    - '/mnt/e/PROJECTS/kaggle_feedback_prize_v1/local_only/NLTK_HOME'
**********************************************************************
```