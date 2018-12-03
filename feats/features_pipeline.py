import logging
import re
from collections import Counter, OrderedDict
import numpy as np
import chardet
from pandas import DataFrame
from html2text import html2text
import os
import functools

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler, PolynomialFeatures, Imputer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.ensemble import RandomTreesEmbedding
import gensim

"""
{
   transforms: [
      {
          type: featurizer, # basic
          field: body_text,
          transforms: [
              {name: tfidf, config: {}}
          ]
      },
      {
          type: compound,  # recursive
          transforms: [{
              type: featurizer,
              field: subject_text,
              transforms: [{name: tfidf, config: {}}]
          }],
          post_process: [{
            name: nmf,
            config: {}
          }]  # none is just concat
      }

   ], # there is a concat prior to post process
   post_processes: [{
       name: svd,
       config: {}
   }]
}

"""

def pipeline_from_config_file(filename):
    """
        public api to access pipeline
        creation when config is a json file
    """
    return pipeline_from_config(json.load(open(filename, 'r')))

def pipeline_from_config(config):
    """
       public api to access pipeline creation
    """
    return _compound_pipeline_from_config(config)

def _transformer_from_config(field, transformer_config):
    name = transformer_config['name']
    configs = transformer_config.get('config', {})
    return get_transformer(name)(field, **configs)    

def _transformers_from_config(field, transformers):
    transforms = [_transformer_from_config(field, transformer) for transformer in transformers]
    return ("%s_features" % field, FeatureUnion(transformer_list=transforms))

def _handle_transform(transform):
    if transform["type"] == "featurizer":
        return _transformers_from_config(transform["field"], transform["transforms"])
    elif transform["type"] == "compound":
        return ('compound', _compound_pipeline_from_config(transform))
    else:
        raise ValueError("invalid transform type: %s" % transform["type"])

def _handle_postprocess(components, post_process):
    name = post_process["name"]
    if name == "recursive":
        steps = post_process["steps"]
        return build_recursive_postprocess(components, steps)
    else:
        configs = post_process.get("config", {})
        return get_postprocess(name)(components, **configs)
    
def _compound_pipeline_from_config(config):
    """
       constructing complex pipelines
    """
    components = [_handle_transform(transform) for transform in config["transforms"]]
    union = ('preprocessed', FeatureUnion(transformer_list=components))

    if "post_process" in config and len(config["post_process"]) > 0:
        post_processed = [_handle_postprocess(Pipeline([union]), post_process) for post_process in config["post_process"]]
    else:
        post_processed = [union]

    out = FeatureUnion(transformer_list=post_processed)
    return Pipeline([('union', out)])

def get_transformer(name):
    """
       some convenience methods for common
       feature creation methods

       todo: handle more manual feature creation pipelines
    """
    transformer_map = {
        "standard_numeric" : build_numeric_column,
        "quantile_numeric" : build_quantile_column,
        "range_numeric"    : build_range_scaler,
        "dummyizer"        : build_dummyizer,
        "null_transformer" : build_null, # don't do anything to this column
        "tfidf"            : build_tfidf_transformer,
        "word_count"       : build_wordcount_transformer,
        "hashing"          : build_feature_hashing_transformer,
        "w2v"              : build_word2vec_transformer
    }
    return transformer_map[name]

def get_postprocess(name):
    """
      todo: handle sequential transforms
    """
    processor_map = {
        "null": build_null,
        "nmf" : build_nmf,
        "svd" : build_svd,
        "lda" : build_lda,
        "rte" : build_rte,
        "poly": build_polynomial,
        "abs" : build_abs,
        "l2"  : build_l2
    }

    return processor_map[name]

"""
   #####################################################################################
   featurizer convenience methods
   #####################################################################################
"""
def build_numeric_column(col):
    return ("numeric_%s" % col, Pipeline([
                ('selector', ItemSelector(col)), 
                ('reshaper', Reshaper()),
                ('floater', Floater()),
                ('scaler', StandardScaler())]))

def build_quantile_column(col,
                          n_quantiles=100):
    return ("quantile_%s" % col, Pipeline([
                ('selector', ItemSelector(col)), 
                ('reshaper', Reshaper()),
                ('quantiler', Quantiler(n_quantiles))]))

def build_range_scaler(col,
                       min=0,
                       max=1):
    return ("min_max %s" % col, Pipeline([
                ('selector', ItemSelector(col)),
                ('reshaper', Reshaper()),
                ('min_max', MinMaxScaler(feature_range=(min, max)))]))

def build_dummyizer(col):
     return ("onehot_s_%s" % col, Pipeline([
           ('selector', ItemSelector(col)),
           ('concat_cols', Concatenator()),
           ('label', Dummyizer())]))

def build_null(col):
    return ("null_%s" % col, Pipeline([
        ('selector', ItemSelector(col)), 
        ('reshaper', Reshaper())]))

def build_feature_hashing_transformer(col,
                                      n_features=2**20,
                                      alternate_sign=True):

    return ("hasher_%s" % col, Pipeline([
        ('selector', ItemSelector(col)),
        ('concat_cols', Concatenator()),
        ('cleaner', WordCleaner()),
        ('hasher', FeatureHasher(input_type="string", n_features=n_features, alternate_sign=alternate_sign))

    ]))

def build_wordcount_transformer(col,
                                binary=False,
                                min_df=0.0,
                                ngrams=2):
    return ("wordcount_%s" % col, Pipeline([
           ('selector', ItemSelector(col)),
           ('concat_cols', Concatenator()),
           ('cleaner', WordCleaner()),
           ('tfidf', CountVectorizer(binary=binary, min_df=min_df, decode_error='ignore', ngram_range=(1,ngrams)))]))

def build_tfidf_transformer(col,
                            min_df=0.0,
                            ngrams=2):
    return ("tfidf_%s" % col, Pipeline([
           ('selector', ItemSelector(col)),
           ('concat_cols', Concatenator()),
           ('cleaner', WordCleaner()),
           ('tfidf', TfidfVectorizer(min_df=min_df, decode_error='ignore', ngram_range=(1,ngrams)))]))

def build_word2vec_transformer(col,
                               replace_pattern=r'[^0-9a-zA-Z ]',
                               rank=100,
                               window=50,
                               workers=1,
                               alpha=0.25,
                               min_count=5,
                               max_vocab_size=None,
                               negative=5,
                               cbow_mean=1,
                               skip_gram=False):
    return ("w2v_%s" % col, Pipeline([
        ('selector', ItemSelector(col)),
        ('concat_cols', Concatenator()),
        ('cleaner', WordCleaner()),
        ('replacer', WordReplacer(replace_pattern=replace_pattern)),
        ('splitter', WordSplitter()),
        ('w2v', W2Vifier(rank=rank,
                         window=window,
                         workers=workers,
                         alpha=alpha,
                         min_count=min_count,
                         max_vocab_size=max_vocab_size,
                         negative=negative,
                         cbow_mean=cbow_mean,
                         skip_gram=skip_gram))
    ]))

"""
   #####################################################################################
   post processor convenience methods
   #####################################################################################
"""
def build_null(pipeline):
    return ('null_pipeline' , pipeline)

def build_polynomial(pipeline,
                     degree=2,
                     interaction_only=False,
                     include_bias=True):
    return ("polynomial", Pipeline([
        ('preprocessed', pipeline),
        ('densinator', Densinator()),
        ('poly', PolynomialFeatures(degree=degree,
                                    interaction_only=interaction_only,
                                    include_bias=include_bias))
    ]))

def build_svd(pipeline,
              rank=50):
    return ("svd", Pipeline([
        ('preprocessed', pipeline),
        ('svd', TruncatedSVD(n_components=rank))
    ]))

def build_rte(pipeline,
              n_estimators=100,
              max_depth=5,
              min_samples_split=2,
              min_samples_leaf=1,
              min_impurity_decrease=0.000001):

    return ("rte", Pipeline([
        ("preprocessed", pipeline),
        ('rte', RandomTreesEmbedding(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease))
    ]))

def build_nmf(pipeline,
              rank=50):

    return ("nmf", Pipeline([
        ("preprocessed", pipeline),
        ("nmf",  NMF(n_components=rank))
    ]))

def build_lda(pipeline,
              rank=50):

    return ("lda", Pipeline([
        ("preprocessed", pipeline),
        ("lda", LatentDirichletAllocation(n_topics=rank))
    ]))

def build_abs(pipeline):
    return ("abs", Pipeline([
        ("preprocessed", pipeline),
        ("abs", AbsoluteValue())
    ]))

def build_l2(pipeline):
    return ("l2", Pipeline([
        ("preprocessed", pipeline),
        ("l2", L2Normalize())
    ]))
    
def build_recursive_postprocess(pipeline, post_process_list):
    if len(post_process_list) > 0:
        process = post_process_list[0]
        processed = Pipeline([_handle_postprocess(pipeline, process)])
        return build_recursive_postprocess(processed, post_process_list[1:])
    else:
        return ('recursed', pipeline)

"""
   #####################################################################################
   custom pipeline components
   #####################################################################################
"""
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if DataFrame is type(X):
            return X[self.key]
        else:
            raise Exception("unsupported itemselector type. implement some new stuff: %s" % type(X))

class Reshaper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:,None]

class Dummyizer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.dummyizer = LabelBinarizer(sparse_output=True)
        self.dummyizer.fit(X)
        return self

    def transform(self, X):
        return self.dummyizer.transform(X)

class Concatenator(BaseEstimator, TransformerMixin):
    def __init__(self, glue=" "):
        self.glue = glue

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = len(list(X.shape))
        out = ["%s" % (self.glue.join(x) if cols > 1 else x) for x in X]
        return out
            
class Floater(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype("float64")

class Densinator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.todense()

class Quantiler(BaseEstimator, TransformerMixin):
    def __init__(self, n_quantiles=100):
        self.n_quantiles = n_quantiles
    def fit(self, X, y=None):
        percentiles = np.linspace(0, 100, self.n_quantiles+2)
        self.quantiles = np.percentile(X, percentiles)
        return self

    def find_quantile(self, x):
        return [1 if self.quantiles[i] < x and self.quantiles[i+1] >= x else 0 for i in range(0, len(self.quantiles) - 1)]
        
    def transform(self, X):
        return [self.find_quantile(x) for x in X]


class WordReplacer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 replace_pattern=r'[^0-9a-zA-Z ]',
                 replacement=""):
        self.replace_pattern = replace_pattern
        self.replacement = replacement

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [re.sub(self.replace_pattern, self.replacement, x) for x in X]

class WordSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, splitter="\s+"):
        self.splitter = splitter

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [re.split(self.splitter, x) for x in X]

class W2Vifier(BaseEstimator, TransformerMixin):
    def __init__(self,
                 rank=100,
                 window=50,
                 workers=1,
                 alpha=0.25,
                 min_count=5,
                 max_vocab_size=None,
                 negative=5,
                 cbow_mean=1,
                 skip_gram=False):
        self.rank           = rank
        self.window         = window
        self.workers        = workers
        self.alpha          = alpha
        self.min_count      = min_count
        self.max_vocab_size = max_vocab_size
        self.negative       = negative
        self.cbow_mean      = cbow_mean
        self.skip_gram      = skip_gram
        self.w2v = None

    def fit(self, X, y=None):
        logging.info("fitting w2v model")
        self.w2v = gensim.models.Word2Vec(X,
                                          size=self.rank,
                                          window=self.window,
                                          workers=self.workers,
                                          alpha=self.alpha,
                                          min_count=self.min_count,
                                          max_vocab_size=self.max_vocab_size,
                                          negative=self.negative,
                                          cbow_mean=self.cbow_mean,
                                          sg=1 if self.skip_gram else 0)
        logging.info("done fitting w2v")
        return self

    def embed(self, x):
        out = functools.reduce(lambda x,y: x+y,
                               [self.w2v.wv[word] for word in x if word in self.w2v.wv],
                               np.zeros(self.rank))
        norm = np.linalg.norm(out)
        if norm > 0:
            return out/norm
        else:
            # vector of zeros (no vocab in model)
            return out
    
    def transform(self, X):
        return [self.embed(x) for x in X]
        

class WordCleaner(BaseEstimator, TransformerMixin):

    def decode(self, content):
        str_bytes = str.encode(content)
        charset = chardet.detect(str_bytes)['encoding']
        return str_bytes.decode(encoding=charset, errors='ignore')

    feature_regex_pipe = [
        (r"\|", " "),
        (r"\r\n?|\n", " "),
        (r"[^\x00-\x7F]+", " "),
        (r"\s+", " "),
        (r"https?://\S+", "_url_"),
        (r"\w{,20}[a-zA-Z]{1,20}[0-9]{1,20}", "_wn_"),
        (r"\d+/\d+/\d+", "_d2_"),
        (r"\d+/\d+", "_d_"),
        (r"\d+:\d+:\d+", "_ts_"),
        (r":", " ")
    ]
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def _text_clean(x):            
            all_clean = html2text(self.decode(x))
            replaced = functools.reduce(lambda acc, re_rep: re.sub(re_rep[0], re_rep[1], acc), self.feature_regex_pipe, all_clean)
            return " ".join([y for y in replaced.split(" ") if len(y) <= 20])

        return map(_text_clean, X)

class AbsoluteValue(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [abs(x) for x in X]

def L2Normalize(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def normalize(self, x):
        norm = np.linalg.norm(x)
        if norm > 0:
            return x/norm
        else:
            return x

    def transform(self, X):
        return [self.normalize(x) for x in X]
