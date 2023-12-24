import csv
import random
import docclass

def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        dataset = [row for row in reader]
    return dataset

def preprocess_dataset(dataset):
    preprocessed_data = []
    for row in dataset:
        title, abstract, topic = row[0], row[1], row[2]
        content = title + ' ' + abstract 
        preprocessed_data.append((content, topic))
    return preprocessed_data

def save_preprocessed_dataset(preprocessed_data, filename='classifierdata.txt'):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("# Fields: Combined Title and Abstract, Topic\n")
        file.write("# Each line represents a document with its associated topic.\n")
        file.write("# The document content is a combination of its title and abstract.\n\n")
        
        for content, topic in preprocessed_data:
            file.write(f"{content}\t{topic}\n")
    print(f"Dataset saved to {filename}")

def set_thresholds(classifier, category_thresholds):
    for category, threshold in category_thresholds.items():
        classifier.setthreshold(category, threshold)

def train_classifier(classifier, training_data):
    for content, topic in training_data:
        classifier.train(content, topic)

def test_classifier(classifier, test_data):
    correct = 0
    for content, topic in test_data:
        predicted = classifier.classify(content)
        if predicted == topic:
            correct += 1
    return correct / len(test_data)

def cross_validation(dataset, classifier_type, category_thresholds=None, k=5):
    random.shuffle(dataset)
    folds = [dataset[i::k] for i in range(k)]
    overall_accuracies = []

    for i in range(k):
        training_data = [item for j, fold in enumerate(folds) if j != i for item in fold]
        test_data = folds[i]

        classifier = classifier_type(docclass.getwords)
        if category_thresholds:
            set_thresholds(classifier, category_thresholds)
        train_classifier(classifier, training_data)
        accuracy = test_classifier(classifier, test_data)
        print(f"Fold {i+1}: Accuracy = {accuracy}")
        overall_accuracies.append(accuracy)

    average_accuracy = sum(overall_accuracies) / len(overall_accuracies)
    print(f"Average Accuracy over {k} folds: {average_accuracy}")
    return average_accuracy

def experiment_with_thresholds(dataset, classifier_type):
    thresholds = {'default': None, 'low': 0.1, 'high': 3.0}
    for threshold_name, threshold_value in thresholds.items():
        print(f"\nExperimenting with {threshold_name} threshold:")
        classifier = classifier_type(docclass.getwords)

        if threshold_value is not None:
            category_thresholds = {topic: threshold_value for topic in set([data[1] for data in dataset])}
            if hasattr(classifier, 'setthreshold'):
                set_thresholds(classifier, category_thresholds)
        else:
            category_thresholds = None

        cross_validation(dataset, classifier_type, category_thresholds)


def main():
    dataset_file = 'AAAI-14_Accepted_Papers_corrected.txt'
    dataset = load_dataset(dataset_file)
    preprocessed_dataset = preprocess_dataset(dataset)
    save_preprocessed_dataset(preprocessed_dataset)

    classifier_choice = input("Choose a classifier (naivebayes/fisher): ").strip().lower()
    classifier_type = docclass.naivebayes if classifier_choice == "naivebayes" else docclass.fisherclassifier

    cross_validation(preprocessed_dataset, classifier_type)
    experiment_with_thresholds(preprocessed_dataset, classifier_type)

if __name__ == "__main__":
    main()
