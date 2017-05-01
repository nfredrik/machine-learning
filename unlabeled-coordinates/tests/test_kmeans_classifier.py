from unittest.mock import patch, MagicMock, ANY, call

from hamcrest import assert_that, is_, all_of, greater_than_or_equal_to, \
    less_than_or_equal_to, equal_to

import numpy as np

from src import ml


def describe_KMeansClassifier():
    def describe__iterate():
        def describe_when_data_classes_around_the_given_centroids_dont_chage():
            @patch('src.ml.classify_points', MagicMock(
                return_value=[
                    [[1, 2], [3, 4]],
                    [[-1, 2], [-3, 4]],
                ]
            ))
            def it_returns_false():
                clf = ml.KMeansClassifier(2)
                clf._clusters = [
                    [[1, 2], [3, 4]],
                    [[-1, 2], [-3, 4]],
                ]

                keep_iterating = clf._iterate(
                    [[1, 2], [3, 4], [-1, 2], [-3, 4]], [[2, 3], [-2, 3]])

                assert_that(keep_iterating, is_(False))

        def describe_when_data_classes_around_the_given_centroids_chage():
            def it_returns_true():
                clf = ml.KMeansClassifier(2)
                clf._clusters = [
                    [[1, 2]],
                    [[3, 4], [-1, 2], [-3, 4]],
                ]

                keep_iterating = clf._iterate(
                    [[1, 2], [3, 4], [-1, 2], [-3, 4]], [[2, 3], [-2, 3]])

                assert_that(keep_iterating, is_(True))

            def it_sets_new_clusters_for_classifer():
                clf = ml.KMeansClassifier(2)
                clf._clusters = []

                clf._iterate([[1, 2], [3, 4], [-1, 2], [-3, 4]],
                            [[2, 3], [-2, 3]])

                assert_that(clf._clusters, is_([
                    [[1, 2], [3, 4]],
                    [[-1, 2], [-3, 4]],
                ]))

    def describe_fit():
        def it_starts_iterating_classification_with_the_random_centroids():
            clf = ml.KMeansClassifier(2)
            clf._iterate = MagicMock(return_value=False)
            clf._random_centroids = MagicMock(return_value=[[0, 0], [-1, 5]])

            clf.fit([[1, 2], [3, 4], [-10, 10]])

            clf._iterate.assert_called_with(ANY, [[0, 0], [-1, 5]])

        @patch('src.ml.centroids_in',
               MagicMock(return_value=[[2, 3], [-5, 5]]))
        def it_recalculates_centroids_with_every_iteration():
            clf = ml.KMeansClassifier(2)
            clf._iterate = MagicMock(side_effect=[True, False])
            clf._random_centroids = MagicMock(return_value=[[0, 0], [-1, 5]])

            clf.fit([[1, 2], [3, 4], [-10, 10]])

            clf._iterate.assert_has_calls([
                call(ANY, [[0, 0], [-1, 5]]),
                call(ANY, [[2, 3], [-5, 5]]),
            ])


def describe_classify_points():
    def it_splits_data_points_to_clusters_using_closest_centroids():
        clusters = ml.classify_points(
            [[-5, 5], [-10, 10], [-6, 6], [5, 5], [6, 6], [10, 10]],
            [[-8, 8], [8, 8]]
        )

        assert_that(clusters, is_([
            [[-5, 5], [-10, 10], [-6, 6]],
            [[5, 5], [6, 6], [10, 10]],
        ]))


def describe_centroids_in():
    def it_returns_mean_point_for_every_cluster():
        centroids = ml.centroids_in([
            [[1.0, 1.0], [1.5, 2.0], [2.0, 3.0]],
            [[10.0, 10.0], [15.0, 20.0], [20.0, 30.0]]
        ])

        assert_that(np.array_equal(centroids, [[1.5, 2.0], [15.0, 20.0]]),
                    is_(True))


def describe_cluster_lists_equal():
    def describe_when_given_cluster_lists_are_different_size():
        def it_returns_false():
            equal = ml.cluster_lists_equal([[1, 2], [3, 4]], [[0, 0]])

            assert_that(equal, is_(False))

    def describe_when_cluster_lists_are_same_size():
        def describe_when_all_clusters_in_lists_are_equal():
            def it_returns_true():
                equal = ml.cluster_lists_equal(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                )

                assert_that(equal, is_(True))

            def xdescribe_when_clusters_are_not_in_the_same_order():
                def it_returns_true():
                    equal = ml.cluster_lists_equal(
                        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                        [[[5, 6], [7, 8]], [[1, 2], [3, 4]]],
                    )

                    assert_that(equal, is_(True))

        def describe_when_cluster_values_dont_match():
            def it_returns_false():
                equal = ml.cluster_lists_equal(
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[[1, 2], [3, 4]], [[5, 6], [10, 11]]],
                )

                assert_that(equal, is_(False))


def describe_random_centroid_in():
    def it_returns_random_point_within_the_given_data_boundaries():
        centroid = ml.random_centroid_in((-15, -3, 20, 10))

        cx = centroid[0]
        cy = centroid[1]
        assert_that(cx, all_of(
            greater_than_or_equal_to(-15), less_than_or_equal_to(20)))
        assert_that(cy, all_of(
            greater_than_or_equal_to(-3), less_than_or_equal_to(10)))


def describe_data_boundaries():
    def it_returns_min_max_x_y():
        min_x, min_y, max_x, max_y = ml.data_boundaries(
            [[-15, 0], [0, 10], [20, -3], [1, 1]])

        assert_that(min_x, is_(-15))
        assert_that(min_y, is_(-3))
        assert_that(max_x, is_(20))
        assert_that(max_y, is_(10))
