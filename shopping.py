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
    # Create dictionary to map month
    months = {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "June": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }

    # Load data from csv file
    with open(filename) as f:

        # Read the csv file
        reader = csv.DictReader(f)

        # Craete a list to store evidence data
        evidence = []

        # Create a list to store labels data
        labels = []

        # Loop through each row to load data to the data set
        for row in reader:

            # Create a list to store evidence data for each row
            data = []

            # Loop through each column except last column in the row to append evidence
            for cell in row:

                # Check the category of column for each evidence data
                if cell in ["Administrative", "Informational", "ProductRelated", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]:

                    # Append to data set with specific type
                    data.append(int(months[row[cell]])) if cell == "Month" else data.append(int(row[cell] == "Returning_Visitor")) if cell == "VisitorType" else data.append(int(row[cell] == "TRUE")) if cell == "Weekend" else data.append(int(row[cell]))

                # Check the category of column for each evidence data
                if cell in ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]:

                    # Append to data set with specific type
                    data.append(float(row[cell]))

            # Append evidence data in row to the evidence list
            evidence.append(data)

            # Append labels data in row to the labels list
            labels.append(int(row["Revenue"] == "TRUE"))

        # Check the length of evidence and labels compare with number of rows from csv file except header
        if len(evidence) != len(labels) != len(reader):

            # Exit system with warning
            sys.exit("Failed to load data")

    # Return evidence and labels data list
    return evidence, labels

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Implementing K-Nearest Neighbor Model
    model = KNeighborsClassifier(n_neighbors=1)

    # Fitting data to model
    model.fit(evidence, labels)

    # Return model after trainning
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Create a varibale to count all posititve labels
    total_positive = 0

    # Create a varibale to count all negative labels
    total_negative = 0

    # Create a variable to count correct positive predictions
    correct_positive = 0

    # Create a variable to count correct negative predictions
    correct_negative = 0

    # Loop through each labels to identify labels
    for i in range(len(labels)):

        # Check the labels whether it's positive or negative
        # If label is positive
        if labels[i] == 1:

            # Increase positive counter by 1
            total_positive += 1

            # Check if model correctly predict or not
            correct_positive += 1 if labels[i] == predictions[i] else 0

        # If label is negative
        elif labels[i] == 0:

            # Increase negative counter by 1
            total_negative += 1

            # Check if model correctly predict or nor
            correct_negative += 1 if labels[i] == predictions[i] else 0

    # Compute the sensitivity
    sensitivity = correct_positive / total_positive

    # Compute the specificity
    specificity = correct_negative / total_negative

    # Return two rates
    return sensitivity, specificity


if __name__ == "__main__":
    main()
