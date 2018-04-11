# iswc_codebase

The 'code' folder contains the python code for Machine Learning classifiers.

The 'data' folder contains the various datasets described in the papre.

The four features mentioned in the paper, 'Classifying Crises-Information Relevancy with Semantics', are Statistical Features (SF), BabelNet Features (SF+SemBN), DBpedia Features (SF+SemDB), and BabelNet+DBpedia Features (SF+SemBNDB).

The four folders, in 'data' folder mentioned hereby, contain the training and test data for each case tested in the paper:

statistical_features: all the train and test files for SF
stat_n_hypernym_en: all the train and test files for SF+SemBN
stat_n_dbpedia_en: all the train and test files for SF+SemDB
stat_n_hypernym_dbpedia_en: all the train and test files for SF+SemBNDB
Folder 'babelfy_labelling' contains Babelfy labels for each tweet across each event. Folder 'expanded_db_semantics_en' contains DBpedia semantics for each Synset extracted via Babelfy labeling.

Folder 'hypernym_en' contains BabelNet semantics (hypernyms) for each Synset and also annotationlemma_hypernym_semantics.csv file that contains entire SemBN for each tweet.