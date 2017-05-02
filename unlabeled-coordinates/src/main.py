import fs
import ml
import metrics
import data


def main() -> None:
    vectors = fs.read_json_from('vectors.json')
    grouped_data = data.vectors_to_arrays(vectors)
    print('Classes in labeled data:', len(grouped_data))

    clf = ml.KMeansClassifier(2)
    ungrouped_vectors = list(map(lambda item: [item['x'], item['y']], vectors))
    clf.fit(ungrouped_vectors)

    print('Classes found with K-Means:', len(clf._clusters))
    print('Classification errors:',
          metrics.classification_errors(grouped_data, clf._clusters))


if __name__ == '__main__':
    main()
