# library utama yang digunakan
import tkinter as tk
from tkinter import *
# library utama yang digunakan
from random import randrange
from typing import List, Any, Union
import tkinter as tk
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


# konversi klasifikasi spesies ke dalam bentuk integer agar mempermudah preprocessing pada knn
def to_int(dataset, column):
    nilai_int = [row[column] for row in dataset]
    unique = set(nilai_int)
    final = dict()
    for i, value in enumerate(unique):
        final[value] = i
        print('[%s] => %d' % (value, i))
    for row in dataset:
        row[column] = final[row[column]]
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
    scores = list()
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

from tkinter.filedialog import askopenfilename

def open_dialog():
    root = Tk()
    # Memanggil class askopenfilename() yang disimpan dalam variable fileName
    # Yang didalamnya terdapat parameter filetypes
    # Didalam filetypes terdapat 2 buah data untuk setting format file yang bisa dibuka
    # Yaitu Python file .py dan all files
    fileName = askopenfilename(filetypes = ( ("Csv file", "*.csv"),("All files", "*.*") ))
    # Mencetak direktori file yang dimasukan kedalam label
    label = Label(root, text="Upload File Berhasi, Silahkan Lanjutkan". fileName)
    label.pack()

# fungsi utama
def main():
    filename = 'iris.data.csv'
    dataset = baca_file(filename)
    for i in range(len(dataset[0]) - 1):
        to_float(dataset, i)
    # konversi spesies ke integer
    to_int(dataset, len(dataset[0]) - 1)

    root = tk.Tk()
    root.geometry('600x400+100+200')
    root.title('Knn Calculator')
    photo = PhotoImage(file="KNN-Algorithm.png")
    label = Label(root, image=photo)
    label.image = photo  # keep a reference!
    label.grid(row=0, column=0, columnspan=20, rowspan=20)

    csv = tk.Label(root, text="Import File csv", width=25, fg="green").grid(row=1, column=0)
    labelNum1 = tk.Label(root, text="Masukkan Nilai K", width=25, fg="green").grid(row=2, column=0)
    labelNum2 = tk.Label(root, text="Masukkan Nilai Permintaan", width=25, fg="green").grid(row=3, column=0)
    labelNum2 = tk.Label(root, text="Masukkan Nilai Persediaan", width=25, fg="green").grid(row=4, column=0)

    labelResult = tk.Label(root)
    labelResult.grid(row=7, column=2)

    entryNum1 = tk.Entry(root, width=30, fg="red")
    entryNum1.grid(row=2, column=2)
    entryNum2 = tk.Entry(root, width=30, fg="red")
    entryNum2.grid(row=3, column=2)
    entryNum3 = tk.Entry(root, width=30, fg="red")
    entryNum3.grid(row=4, column=2)
    buttoncsv = tk.Button(root, text='Upload File', width=25, command=open_dialog)
    buttoncsv.grid(row=1, column=2)

    tot_neighbors = entryNum1.get()
    val1 = entryNum2.get()
    val2 = entryNum3.get()
    test_data = [val1, val2, 0, 0]
    klasifikasi = klasifikasi_prediksi(dataset, test_data, tot_neighbors)
    errors = evaluate_algorithm(dataset, k_accuracy, tot_neighbors, tot_neighbors)

    button = tk.Button(root, text='Hitung Prediksi', width=20, command=klasifikasi).grid(row=3, column=3)
    button = tk.Button(root, text='Hitung Prosentase Error', width=20, command=errors).grid(row=3, column=3)
    button = tk.Button(root, text='Tambilkan Sebaran Data', width=20).grid(row=3, column=3)
    # call_result = partial(call_result, labelResult, number1, number2)
    #
    # buttonCal = tk.Button(root, text="Calculate", command=call_result).grid(row=3, column=0)

    root.mainloop()


# recall fungsi main()
if __name__ == "__main__":
    main()
