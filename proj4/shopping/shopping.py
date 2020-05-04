import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    df = []

    with open(filename, 'r') as f:
        df_reader = csv.reader(f)
        next(df_reader)

        for record in df_reader:
            df.append(record)

    for record in df:
        #Converting Month to an index from 0 (January) to 11 (December)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, month in enumerate(months):
            if record[10] == month:
                record[10] = i
                break

        #Converting VisitorType to an int
        if record[15] == 'Returning_Visitor':
            record[15] = 1
        else:
            record[15] = 0
        
        #Converting Weekend to an int
        if record[16] == 'TRUE':
            record[16] = 1
        elif record[16] == 'FALSE':
            record[16] = 0

        #Converting labels (Revenue) to an int
        if record[17] == 'TRUE':
            record[17] = 1
        elif record[17] == 'FALSE':
            record[17] = 0

        #Converting data to appropriate data types
        for i in range(18):
            if i in [1, 3, 5, 6, 7, 8, 9]:
                record[i] = float(record[i])
            else:
                record[i] = int(record[i])

    evidence = []
    labels = []

    for record in df:
        evidence.append(record[:17])
        labels.append(record[17])

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    pos_count = 0
    pos_correct = 0
    neg_count = 0
    neg_correct = 0

    for prediction, label in zip(predictions, labels):
        if label == 1:
            pos_count += 1
            if prediction == 1:
                pos_correct += 1
        elif label == 0:
            neg_count += 1
            if prediction == 0:
                neg_correct += 1

    sensitivity = pos_correct/pos_count
    specificity = neg_correct/neg_count

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
