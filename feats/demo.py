import sys
import os
import logging
import pandas as pd
import functools

from annoy import AnnoyIndex


from feats.features_pipeline import pipeline_from_config

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, '..', 'data', 'Damage Tracker.xlsx')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', stream=sys.stdout, level=logging.INFO)
    logging.info("reading excel file")
    df = pd.read_excel(file_path, header=0, index_col='ID')
    logging.info("input shape:")
    print (df.shape)

    config_one = {
        # a list of transforms on different columns
        "transforms" : [
            {
                "type"       : "featurizer", # basic column transformation
                "field"      : "Damaged Address:",
                "transforms" : [{
                    "name"   : "word_count"
                }]
            },
            {
                "type"       : "featurizer",
                "field"      : "City",
                "transforms" : [{
                    "name"   : "dummyizer"
                }]
            },
            {
                "type"       : "featurizer",
                "field"      : "Service Center:",
                "transforms" : [{
                    "name"   : "dummyizer"
                }]
            },
            {
                "type"       : "compound", # recusrive pipeline
                "transforms" : [
                    {
                        "type"       : "featurizer",
                        "field"      : "Damaged Address:",
                        "transforms" : [{"name": "tfidf"}]
                    }
                ],
                "post_process" : [
                    {"name" : "svd"},
                    {"name" : "lda"}
                ]
            }
        ]
    }

    logging.info("building pipeline")
    pipeline = pipeline_from_config(config_one)

    logging.info("fitting pipeline on dataframe")
    vectors = pipeline.fit_transform(df)

    logging.info("output shape")
    print (vectors.shape)


    logging.info("todo: something more useful with this data")
