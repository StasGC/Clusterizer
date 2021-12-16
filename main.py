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


class TextNormalizer:
    def __init__(self):
        pass

    def _normalize(self, text):
        raise AttributeError('Not implemented yet.')

    def fit(self, df, feature):
        return self

    def transform(self, df, feature):
        if df is None:
            return df

        for i in range(len(df[feature])):
            df[feature].iloc[i] = self._normalize(df[feature].iloc[i])

        return df

    def fit_transform(self, df, feature):
        self.fit(df, feature)

        return self.transform(df, feature)


class StopWordsRemover(TextNormalizer):
    def _normalize(self, text):
        if text is None:
            return ''

        stopwords_list = stopwords.words('russian')

        text = text.lower()
        # strip html
        pattern1 = re.compile(r'<.*?>')
        pattern2 = re.compile(r'\t|\n|\r')
        pattern3 = re.compile(r'\\+n|\\+t|\\+r')
        pattern4 = re.compile(r'\s\S\s')

        text = pattern1.sub('', text)
        text = pattern2.sub('', text)
        text = pattern3.sub('', text)

        text = re.sub('-', '', text)
        text = re.sub('_', '', text)
        # Remove data between square brackets
        text = re.sub('\[[^]]*\]', '', text)
        # removes punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[0-9]+', ' ', text)

        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)

        text = pattern4.sub(' ', text)

        text = text.strip()

        words = [word for word in nltk.word_tokenize(text) if word not in stopwords_list
                 and word != ' '
                 and word.strip() not in punctuation]

        return ' '.join(words)


class Lemmatizer(TextNormalizer):
    def __init__(self, stemmer=Mystem()):
        super().__init__()
        self.stemmer = stemmer

    def _normalize(self, text):
        if text is None:
            return ''

        lemmas = self.stemmer.lemmatize(text)
        words = [lemma for lemma in lemmas if lemma != '\n']

        return ' '.join(words)


class Stemmer(TextNormalizer):
    def __init__(self, stemmer=Mystem()):
        super().__init__()
        self.stemmer = stemmer

    def _normalize(self, text):
        if text is None:
            return ''

        words = [self.stemmer.stem(word) for word in nltk.word_tokenize(text)]
        words = [word for word in words if word != '\n']

        return ' '.join(words)


class MyTfIdfVectorizer:
    def __init__(self):
        self.tf_idf_transformer = TfidfVectorizer()

    def fit(self, df, feature):
        return self

    def transform(self, df, feature):
        tf_idf_new_features = self.tf_idf_transformer.fit_transform(df[feature])
        tf_idf_norm_new_features = normalize(tf_idf_new_features)
        tf_idf_array = tf_idf_norm_new_features.toarray()

        df = df.drop(columns=[feature])
        return pd.concat([
            df,
            pd.DataFrame(tf_idf_array),
        ], axis=1)

    def fit_transform(self, df, feature):
        self.fit(df, feature)

        return self.transform(df, feature)


class FeatureGenerator:
    def __init__(self, ohe_features=None, le_features=None, num_features=None, delete_features=None, scale=True):
        self._ohe_features = ohe_features or []
        self._le_features = le_features or []
        self._num_features = num_features or []
        self._delete_features = delete_features or []
        self.scale = scale

    def fit(self, df):
        return self

    def transform(self, df):
        if self.scale:
            scaler = StandardScaler()
        else:
            scaler = 'passthrough'

        if self._num_features:
            num_transformer = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', scaler),
                ]
            )
        else:
            num_transformer = 'passthrough'

        if self._ohe_features:
            ohe_transformer = OneHotEncoder(handle_unknown="ignore")
        else:
            ohe_transformer = 'passthrough'

        if self._le_features:
            le_transformer = OrdinalEncoder()
        else:
            le_transformer = 'passthrough'

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self._num_features),
                ('bin', le_transformer, self._le_features),
                ('cat', ohe_transformer, self._ohe_features),
            ]
        )

        new_features = preprocessor.fit_transform(df)
        df_new_features = pd.DataFrame(new_features)

        all_features = self._num_features + self._ohe_features + self._le_features + self._delete_features
        df = df.drop(columns=all_features)

        return pd.concat([
            df,
            df_new_features,
        ], axis=1)

    def fit_transform(self, df):
        self.fit(df)

        return self.transform(df)


class UpdateTextFeatures:
    def __init__(self, text_features):
        self._text_features = text_features
        self.steps = None

    def with_steps(self, steps):
        self.steps = steps
        return self

    def fit(self, df):
        return self

    def transform(self, df):
        if self.steps is None:
            raise AttributeError('Parameter steps is empty. Call method with_steps to fill it.')

        for feature in self._text_features:
            for transformer in self.steps:
                df = transformer.fit_transform(df, feature)

        return df

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
        else:
            raise AttributeError('This type of algorithm is not supported.')

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
