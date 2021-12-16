# Программа для матчинга дупликатов

Данная программа реализована для матчинга дупликатов в наборе входных данных. 

В коде представлены классы для обработки датасета, а также класс с алгоритмами кластеризации:
- Clusterizer: класс, в котором реализованы алгоритмы кластеризации. На данный момент доступны следующие алгоритмы: KMeans(`k-means`), GaussianMixture(`GMM`), AgglomerativeClustering(`Agg`).

Данным приложением можно легко воспользоваться, подключив `MyPipeline`, который собирает все компоненты вместе, последовательно выполняет их, прогоняя через них данные, и выдает финальный результат -- вектор *prediction*.

Чтобы посмотреть на близкие друг к другу объявления, можно воспользоваться функцией `get_duplicate_from_prediction`.
