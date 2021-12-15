import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.preprocessing import normalize

nltk.download('punkt')
nltk.download("stopwords")

from string import punctuation
from pymystem3 import Mystem

import re


class DataPreprocessor:
    mystem = None

    def __init__(self, features, lemming=False, stemming=False):
        self.features = features
        self.lemming = lemming
        self.stemming = stemming

    def _lemmatize(self, text):
        if not self.mystem:
            self.mystem = Mystem()

        lemmas = self.mystem.lemmatize(text)
        words = [lemma for lemma in lemmas if lemma != '\n']

        return ' '.join(words)

    def _stemm(self, text):
        stemmer = PorterStemmer()

        words = [stemmer.stem(word) for word in nltk.word_tokenize(text)]
        words = [word for word in words if word != '\n']

        return ' '.join(words)

    def clean_column(self, data):
        if data is None:
            return ''

        stopwords_list = stopwords.words('russian')

        data = data.lower()
        # strip html
        pattern1 = re.compile(r'<.*?>')
        pattern2 = re.compile(r'\t|\n|\r')
        pattern3 = re.compile(r'\\+n|\\+t|\\+r')
        pattern4 = re.compile(r'\s\S\s')

        data = pattern1.sub('', data)
        data = pattern2.sub('', data)
        data = pattern3.sub('', data)

        data = re.sub('-', '', data)
        data = re.sub('_', '', data)
        # Remove data between square brackets
        data = re.sub('\[[^]]*\]', '', data)
        # removes punctuation
        data = re.sub(r'[^\w\s]', ' ', data)
        data = re.sub(r'[0-9]+', ' ', data)

        data = re.sub(r"\'ve", " have ", data)
        data = re.sub(r"can't", "cannot ", data)
        data = re.sub(r"n't", " not ", data)
        data = re.sub(r"I'm", "I am", data)
        data = re.sub(r" m ", " am ", data)
        data = re.sub(r"\'re", " are ", data)
        data = re.sub(r"\'d", " would ", data)
        data = re.sub(r"\'ll", " will ", data)

        data = pattern4.sub(' ', data)

        data = data.strip()

        if self.lemming:
            data = self._lemmatize(data)

        if self.stemming:
            data = self._stemm(data)

        words = [word for word in nltk.word_tokenize(data) if word not in stopwords_list
                 and word != ' '
                 and word.strip() not in punctuation]

        data = ' '.join(words)
        return data

    def fit(self, df):
        return self

    def transform(self, df):
        for feature in self.features:
            for i in range(len(df[feature])):
                df[feature].iloc[i] = self.clean_column(df[feature].iloc[i])

        return df

    def fit_transform(self, df):
        self.fit(df)

        return self.transform(df)


class FeatureGenerator:
    _ohe_features = (
        'condition',
        'preferred_way_to_contact',
    )

    _le_features = (
        'origin',
    )

    _num_features = (
        'price',
        'latitude',
        'longitude',
    )

    _tf_idf_features = (
        'title',
        'description',
    )

    def __init__(self, scale=True):
        self.scale = scale

    def _tf_idf(self, data, feature):
        tf_idf_transformer = TfidfVectorizer()

        _feature = 0 if feature == 'title' else 1

        tf_idf_new_features = tf_idf_transformer.fit_transform(data[self._tf_idf_features[_feature]])
        tf_idf_norm_new_features = normalize(tf_idf_new_features)
        tf_idf_array = tf_idf_norm_new_features.toarray()

        return pd.DataFrame(tf_idf_array)

    def fit(self, data):
        return self

    def transform(self, data):
        if self.scale:
            scaler = StandardScaler()
        else:
            scaler = 'passthrough'

        data['condition'].fillna('No', inplace=True)
        data['preferred_way_to_contact'].fillna('No', inplace=True)

        num_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', scaler),
            ]
        )
        ohe_transformer = OneHotEncoder(handle_unknown="ignore")
        le_transformer = OrdinalEncoder()
        text_feature_transformer = DataPreprocessor(self._tf_idf_features)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self._num_features),
                ('bin', le_transformer, self._le_features),
                ('cat', ohe_transformer, self._ohe_features),
            ]
        )

        data = text_feature_transformer.fit_transform(data)

        new_features = preprocessor.fit_transform(data)
        df_new_features = pd.DataFrame(new_features)

        df_title = self._tf_idf(data, 'title')
        df_description = self._tf_idf(data, 'description')

        return pd.concat([
            df_new_features,
            df_title,
            df_description
        ], axis=1)

    def fit_transform(self, df):
        self.fit(df)

        return self.transform(df)


class Clusterizer:
    def __init__(
            self,
            model='k-means',
            n_components=5,
            max_iter=10,
            algorithm='auto',
            random_state=42,
            linkage='ward',
    ):
        self.model = model
        self.n_components = n_components
        self.max_iter = max_iter
        self.algorithm = algorithm
        self.random_state = random_state
        self.linkage = linkage

    def fit(self, data):
        return self

    def predict(self, data):
        train_data = data.copy()

        n_clusters = int(train_data.shape[0] / 2)
        algo = 'passthrough'
        if self.model == 'k-means':
            algo = KMeans(
                n_clusters=n_clusters,
                max_iter=self.max_iter,
                algorithm=self.algorithm,
                random_state=self.random_state
            )
        elif self.model == 'GMM':
            algo = GaussianMixture(
                n_components=n_clusters,
            )
        elif self.model == 'Agg':
            algo = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.linkage,
            )

        pca = PCA(n_components=self.n_components)

        pipeline = Pipeline(
            steps=[
                ('pca', pca),
                ('clusterization', algo),
            ]
        )
        pipeline.fit(train_data)
        prediction = pipeline.predict(train_data)

        return prediction


class MyPipeline:
    def __init__(self, steps):
        self.steps = steps

    def __len__(self):
        return len(self.steps)

    @property
    def _final_estimator(self):
        estimator = self.steps[-1][1]
        return 'passthrough' if estimator is None else estimator

    def _validate_transformers(self):
        last_name, estimator = self.steps[-1]

        for name, transformer in self.steps[:-1]:
            if transformer is None or transformer == 'passthrough':
                continue

            if not (hasattr(transformer, 'fit') or hasattr(transformer, 'fit_transform')) \
                    or not hasattr(transformer, 'transform'):
                raise TypeError(
                    'Transformers should have fit and transform method.\n' +
                    f'Transformer with name {name} have not fit or transform'
                )

        if (
                estimator is not None
                and estimator != 'passthrough'
                and not (hasattr(estimator, 'fit') or hasattr(estimator, 'predict'))
        ):
            raise TypeError('Estimator should have fit and predict method')

    def fit(self, X):
        self._validate_transformers()

        data = X.copy()

        for idx, (name, transformer) in enumerate(self.steps[:-1]):
            if transformer == 'passthrough':
                continue

            data, fitted_transformer = _fit_transform_one(transformer, data)
            self.steps[idx] = (name, fitted_transformer)

        return data

    def transform(self, X):
        estimator = self._final_estimator

        if estimator == 'passthrough':
            return X

        estimator.fit(X)
        prediction = estimator.predict(X)

        return prediction

    def fit_transform(self, X):
        data = self.fit(X)

        return self.transform(data)


def _fit_transform_one(transformer, X):
    X_transformed = X
    fitted_transformer = transformer

    if hasattr(transformer, 'fit_transform'):
        X_transformed = transformer.fit_transform(X)
    elif hasattr(transformer, 'fit'):
        fitted_transformer = transformer.fit(X)

    return X_transformed, fitted_transformer


def get_duplicate_from_prediction(prediction, data):
    duplicate_df = {
        'id1': [],
        'id2': [],
    }

    clusters = set(prediction)
    for cluster in clusters:
        matches = np.where(prediction == cluster)[0]

        first = data.loc[matches[0]]['id']
        for j in range(1, len(matches)):
            duplicate_df['id1'].append(first)
            duplicate_df['id2'].append(data.loc[matches[j]]['id'])

    return pd.DataFrame(duplicate_df)
