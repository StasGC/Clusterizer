# Программа для матчинга дупликатов

Данная программа реализована для матчинга дупликатов в наборе входных данных. 

В коде представлены классы для обработки датасета, а также класс с алгоритмами кластеризации:
- Clusterizer: класс, в котором реализованы алгоритмы кластеризации. На данный момент доступны следующие алгоритмы: KMeans(`k-means`), GaussianMixture(`GMM`), AgglomerativeClustering(`Agg`).

Данным приложением можно легко воспользоваться, подключив `MyPipeline`, который собирает все компоненты вместе, последовательно выполняет их, прогоняя через них данные, и выдает финальный результат -- вектор *prediction*.

Чтобы посмотреть на близкие друг к другу объявления, можно воспользоваться функцией `get_duplicate_from_prediction`.

# Quick Start

Более подробный пример можно посмотреть в файле `Example.ipynb`.

    from main import *

    df = pd.read_csv('dataset.csv')
    
	_ohe_features = [
	    'condition',
	    'preferred_way_to_contact',
	]

	_le_features = [
	    'origin',
	]

	_num_features = [
	    'price',
	    'latitude',
	    'longitude',
	]

	_tf_idf_features = [
	    'title',
	    'description',
	]

	_delete_features = [
	     'id',
	     'expired_at',
	     'created_at',
	     'published_at',
	     'snapshot_date',
	     'category_id',
	     'anonymized_user_id',
	     'status',
	     'photo_hash',
	     'region_id',
	     'region_name',
	]

	pipeline = MyPipeline(
	    steps=[
		('text_preprocessor', UpdateTextFeatures(_tf_idf_features).with_steps(
			steps=[
			    StopWordsRemover(),
			    Lemmatizer(),
			    MyTfIdfVectorizer(),
			]
		    )
		),
		('feature_generation', FeatureGenerator(_ohe_features, _le_features, _num_features, _delete_features)),
		('clusterization', Clusterizer())
	    ]
	)

    pred = pipeline.fit_transform(dataset)
    get_duplicate_from_prediction(pred, dataset)
