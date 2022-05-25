# Awakened at CheckThat! 2022: Fake News Detection using BiLSTM and Sentence Transformer

### Packages:
- tensorflow-gpu
- keras
- pandas
- scipy
- numpy
- sentence_transformers

### Create sentence embeddings:

``python create_embeddings.py FIN_TRAIN FIN_DEV FIN_TEST DIR_NAME LANG``

Where:
- FIN_TRAIN - the training file 
- FIN_DEV - the dev file
- FIN_TEST - the test file
- DIR_NAME - the directory name for the output files that contain the embeddings (use different directories for each language)
- LANG - lanauge name, i.e., ``en`` or ``de``


### Create the mono-lingual English model

``python model_en.py DIR_NAME_EN``

Where:
- DIR_NAME_EN - name of the directory where output files for English where created using the ``create_embeddings.py`` script

### Create the cross-lingual English to German model

``python model_en.py DIR_NAME_DE``

Where:
- DIR_NAME_DE - name of the directory where output files for German where created using the ``create_embeddings.py`` script
