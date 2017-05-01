from hamcrest import assert_that, is_

from src.metrics import cluster_matches, cluster_map, classification_errors


def describe_cluster_matches():
    def it_returns_number_of_points_matching_in_given_clusters():
        matches = cluster_matches(
            [[1, 2], [3, 4], [5, 6]],
            [[5, 6], [-3, 10], [1, 2]]
        )

        assert_that(matches, is_(2))


def describe_cluster_map():
    def it_returns_dictionary_mapping_between_two_cluster_lists_by_the_count_of_matching_points():
        cl_map = cluster_map(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[-1, 2], [-3, 4], [-5, 6]],
            ],
            [
                [[-1, 2], [3, 4], [5, 6]],
                [[-3, 4], [-5, 6], [1, 2]],
            ],
        )

        assert_that(cl_map, is_({0: 0, 1: 1}))


def describe_classification_errors():
    def describe_when_given_clusters_are_in_the_same_order():
        def it_returns_count_of_mismatching_vectors_in_the_clusters():
            errors = classification_errors(
                [
                    [[1, 2], [3, 4], [5, 6]],
                    [[-1, 2], [-3, 4], [-5, 6]],
                ],
                [
                    [[-1, 2], [3, 4], [5, 6]],
                    [[-3, 4], [-5, 6], [1, 2]],
                ],
            )

            assert_that(errors, is_(2))

    def describe_when_given_clusters_are_out_of_order():
        def it_finds_the_right_clusters_to_compare_and_returns_count_of_mismatching_vectors_in_the_clusters():
            errors = classification_errors(
                [
                    [[1, 2], [3, 4], [5, 6]],
                    [[-1, 2], [-3, 4], [-5, 6]],
                    [[-1, -1], [-2, -2], [-3, -3]],
                ],
                [
                    [[-1, -1], [-2, -2], [-3, -3]],
                    [[-3, 4], [-5, 6], [1, 2]],
                    [[-1, 2], [3, 4], [5, 6]],
                ],
            )

            assert_that(errors, is_(2))
