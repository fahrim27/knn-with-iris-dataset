# library utama yang digunakan
from random import randrange
from typing import List, Any, Union

import pandas as pd
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns


# inisiasi pembacaan file dataset csv
def baca_file(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Konversi data integer ke float untuk memudahkan proses training data
def to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def to_int(dataset, column):
    nilai_int = [row[column] for row in dataset]
    unique = set(nilai_int)
    final = dict()
    print(
        "=================================================Data Prediksi==============================================")
    for i, value in enumerate(unique):
        final[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = final[row[column]]
    print(
        "============================================================================================================")
    return final


# membaca sebaran nilai dari kolom dataset yang di uji
def dataset_minmax_column(dataset):
    minmax_column = list()
    for i in range(len(dataset[0])):
        nilai_column = [row[i] for row in dataset]
        nilai_min = min(nilai_column)
        nilai_max = max(nilai_column)
        minmax_column.append([nilai_min, nilai_max])
    return minmax_column


# fungsi normalisasi dataset untuk menghilangkan noise data, jika ada
def normalisasi_dataset(dataset, minmax_column):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax_column[i][0]) / (minmax_column[i][1] - minmax_column[i][0])


# fungsi untuk menemukan euclidean distance dari row data training dan data test
def euclidean_distance(row1, row2):
    jarak = 0.0
    for i in range(len(row1) - 1):
        jarak += (row1[i] - row2[i]) ** 2
    return sqrt(jarak)


# fungsi pencarian tetangga terdekat
def Kneighbors(train, test_row, num_neighbors):
    jarak = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        jarak.append((train_row, dist))
    jarak.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(jarak[i][0])
    return neighbors


# fungsi prediksi klasifikasi spesies bunga berdasarkan hasil pengelahan knn
def klasifikasi_prediksi(train, test_data, tot_neighbors):
    neighbors = Kneighbors(train, test_data, tot_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Kalkulasi persentasi akurasi dari klasifikasi knn
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def k_accuracy(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = klasifikasi_prediksi(train, row, num_neighbors)
        predictions.append(output)
    return predictions


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores: List[Union[float, Any]] = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# fungsi utama
def main():
    filename = 'iris.data.csv'
    dataset = baca_file(filename)
    for i in range(len(dataset[0]) - 1):
        to_float(dataset, i)
    # konversi spesies ke integer
    to_int(dataset, len(dataset[0]) - 1)

    # input data k
    print(
        "\n====================================================Inisiasi input dari "
        "User=====================================")
    neighbors_input = input("Masukan nilai K-tetangga = ")
    tot_neighbors: int = int(neighbors_input)

    # input data yang akan di test dengan training data utama
    print(
        "\n====================================================Masukkan data "
        "Testing========================================")
    sepal_length = input("Masukan nilai Sepal Length = ")
    sepal_width = input("Masukan nilai Sepal Width = ")
    petal_length = input("Masukan nilai Petal Length = ")
    petal_width = input("Masukan nilai Petal Width = ")
    test_data = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]

    # prediksi label dari hasil proses knn
    print(
        "\n====================================================Hasil Presiksi "
        "KNN===========================================")
    label = klasifikasi_prediksi(dataset, test_data, tot_neighbors)
    print('data test=%s, Hasil Prediksi= %s' % (test_data, label))

    scores = evaluate_algorithm(dataset, k_accuracy, tot_neighbors, tot_neighbors)
    print('Hasil Confusin Matrix pada Data Prediksi: %s' % scores)
    print('Tingkat Akurasi Prediksi: %.3f%%' % (sum(scores) / float(len(scores))))

    # tampilkan informasi tetangga terdekat sebagai validasi hasil
    print(
        "\n=================================================Nilai Tetang "
        "Terdekat========================================")
    print("Array Sepal dan Petal Tetangga: ", Kneighbors(dataset, test_data, tot_neighbors))
    nilaiKneighbors = Kneighbors(dataset, test_data, tot_neighbors)
    for i in range(len(nilaiKneighbors)):
        for j in range(len(nilaiKneighbors[i])):
            print(nilaiKneighbors[i][j], end=' ')
        print()
    print(
        "====================================Jenis Spesies dari Tetangga Terdekat=====================================")
    neigh_value = [row[-1] for row in Kneighbors(dataset, test_data, tot_neighbors)]
    print(neigh_value)
    sumNeigh = 0
    for i in neigh_value:
        sumNeigh = sumNeigh + i
        print("Iris-Versicolor")

    # visualisasi iris dataset
    iris = load_iris()
    x_index = 0
    y_index = 1
    formatter = plt.FuncFormatter(lambda j, *args: iris.target_names[int(j)])
    plt.figure(figsize=(5, 4))
    plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])
    plt.tight_layout()
    plt.show()

    sns.set(style="ticks", color_codes=True)
    iris_new = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw'
                           '/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv')
    sns.pairplot(iris_new)
    plt.show()

    iris_new['Length_ratio'] = iris_new['petal_length'] / iris_new['sepal_width']
    iris_new['Width_ratio'] = iris_new['sepal_length'] / iris_new['petal_width']
    sns.jointplot(x='Length_ratio', y='Width_ratio', data=iris_new)
    plt.show()


# recall fungsi main()
if __name__ == "__main__":
    main()
