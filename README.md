# Программа для матчинга дупликатов

Данная программа реализована для матчинга дупликатов в наборе входных данных. 

В коде представлены 3 класса:
- DataPreprocessor: обработка текстовых значений
- FeatureGeneration: генерация фич из исходных данных для алгоритмов кластеризации
- Clusterizer: класс, в котором реализованы алгоритмы кластеризации. На данный момент доступны следующие алгоритмы: KMeans(`k-means`), GaussianMixture(`GMM`), AgglomerativeClustering(`Agg`).

Данным приложением можно легко воспользоваться, подключив `MyPipeline`, который собирает все компоненты вместе, последовательно выполняет их, прогоняя через них данные, и выдает финальный результат -- вектор *prediction*.

Чтобы посмотреть на близкие друг к другу объявления, можно воспользоваться функцией `get_duplicate_from_prediction`.

# Quick Start

Более подробный пример можно посмотреть в файле `Example.ipynb`.

    from main import *
	
    df = pd.read_csv('dataset.csv')
    _tf_idf_features = (
        'title',
        'description',
    )

    pipeline = MyPipeline(
        steps=[
            ('preprocessor', DataPreprocessor(_tf_idf_features)),
            ('feature_generation', FeatureGenerator()),
            ('clusterization', Clusterizer())
        ]
    )

    pred = pipeline.fit_transform(dataset)
    get_duplicate_from_prediction(pred, dataset)
