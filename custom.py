import re
import pickle
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def sanitize_column_names(columns):
    """Remove all non-alphanumeric characters from column names."""
    return [re.sub('[^0-9a-zA-Z]+', '_', col) for col in columns]

def load_model(code_dir):
    with open(f"{code_dir}/model_artifacts/pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open(f"{code_dir}/model_artifacts/model.pkl", "rb") as f:
        clf = pickle.load(f)
    return (pipeline, clf)

def score(data, model, **kwargs):
    data.columns = sanitize_column_names(data.columns)

    pipeline, clf = model
    for target_col in ["target_binary"]:
        if target_col in data:
            data.pop(target_col)
    
    transformed_csr = pipeline.transform(data) 
    predictions = pd.DataFrame(clf.predict_proba(transformed_csr))
    predictions.columns = ["FALSE", "TRUE"]
    return predictions