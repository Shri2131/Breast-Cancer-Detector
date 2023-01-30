# Importing Libraries
import tkinter.messagebox as TkMessageBox
import tkinter
from tkinter import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

mydf = pd.read_csv("data.csv")

# Defining Dependent and Independent Variables
X = mydf.iloc[:, 2:32].values  # Independent Variable
Y = mydf.iloc[:, 1]  # Dependent Variable

# Splitting in training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=2)

# Training the data
model = LogisticRegression(max_iter=3000)
model.fit(X_train, Y_train)
print("Model Created Successfully")

# Predicting Values
Y_pred_test = model.predict(X_test)
Y_pred_train = model.predict(X_train)

# Defining Variables
root = Tk()
A1 = DoubleVar()
A2 = DoubleVar()
A3 = DoubleVar()
A4 = DoubleVar()
A5 = DoubleVar()
A6 = DoubleVar()
A7 = DoubleVar()
A8 = DoubleVar()
A9 = DoubleVar()
A10 = DoubleVar()
A11 = DoubleVar()
A12 = DoubleVar()
A13 = DoubleVar()
A14 = DoubleVar()
A15 = DoubleVar()
A16 = DoubleVar()
A17 = DoubleVar()
A18 = DoubleVar()
A19 = DoubleVar()
A20 = DoubleVar()
A21 = DoubleVar()
A22 = DoubleVar()
A23 = DoubleVar()
A24 = DoubleVar()
A25 = DoubleVar()
A26 = DoubleVar()
A27 = DoubleVar()
A28 = DoubleVar()
A29 = DoubleVar()
A30 = DoubleVar()

# GUI


def fun():
    LA = Label(root, text="Testing Data Accuracy").grid(row=17, column=1)
    LB = Label(root, text=metrics.accuracy_score(
        Y_test, Y_pred_test)*100).grid(row=17, column=2)
    LC = Label(root, text="Training Data Accuracy").grid(row=18, column=1)
    LD = Label(root, text=metrics.accuracy_score(
        Y_train, Y_pred_train)*100).grid(row=18, column=2)
    L1 = Label(root, text="Radius Mean").grid(row=1, column=1)
    w1 = Entry(root, bd=5, textvariable=A1).grid(row=1, column=2)
    L2 = Label(root, text="Texture Mean").grid(row=2, column=1)
    w2 = Entry(root, bd=5, textvariable=A2).grid(row=2, column=2)
    L3 = Label(root, text="Perimeter Mean").grid(row=3, column=1)
    w3 = Entry(root, bd=5, textvariable=A3).grid(row=3, column=2)
    L4 = Label(root, text="Area Mean").grid(row=4, column=1)
    w4 = Entry(root, bd=5, textvariable=A4).grid(row=4, column=2)
    L5 = Label(root, text="Smoothness Mean").grid(row=5, column=1)
    w5 = Entry(root, bd=5, textvariable=A5).grid(row=5, column=2)
    L6 = Label(root, text="Compactness  Mean").grid(row=6, column=1)
    w6 = Entry(root, bd=5, textvariable=A6).grid(row=6, column=2)
    L7 = Label(root, text="Concavity Mean").grid(row=7, column=1)
    w7 = Entry(root, bd=5, textvariable=A7).grid(row=7, column=2)
    L8 = Label(root, text="Concave Points Mean").grid(row=8, column=1)
    w8 = Entry(root, bd=5, textvariable=A8).grid(row=8, column=2)
    L9 = Label(root, text="Symmetry Mean").grid(row=9, column=1)
    w9 = Entry(root, bd=5, textvariable=A9).grid(row=9, column=2)
    L10 = Label(root, text="Fractal Dimension Mean").grid(row=10, column=1)
    w10 = Entry(root, bd=5, textvariable=A10).grid(row=10, column=2)
    L11 = Label(root, text="Radius se").grid(row=11, column=1)
    w11 = Entry(root, bd=5, textvariable=A11).grid(row=11, column=2)
    L12 = Label(root, text="Texture se").grid(row=12, column=1)
    w12 = Entry(root, bd=5, textvariable=A12).grid(row=12, column=2)
    L13 = Label(root, text="Perimeter se").grid(row=13, column=1)
    w13 = Entry(root, bd=5, textvariable=A13).grid(row=13, column=2)
    L14 = Label(root, text="Area se").grid(row=14, column=1)
    w14 = Entry(root, bd=5, textvariable=A14).grid(row=14, column=2)
    L15 = Label(root, text="Smoothness se").grid(row=15, column=1)
    w15 = Entry(root, bd=5, textvariable=A15).grid(row=15, column=2)
    L16 = Label(root, text="Compactness se").grid(row=1, column=3)
    w16 = Entry(root, bd=5, textvariable=A16).grid(row=1, column=4)
    L17 = Label(root, text="Concavity se").grid(row=2, column=3)
    w17 = Entry(root, bd=5, textvariable=A17).grid(row=2, column=4)
    L18 = Label(root, text="Concave Point se").grid(row=3, column=3)
    w18 = Entry(root, bd=5, textvariable=A18).grid(row=3, column=4)
    L19 = Label(root, text="Symmetry se").grid(row=4, column=3)
    w19 = Entry(root, bd=5, textvariable=A19).grid(row=4, column=4)
    L20 = Label(root, text="Fractal Dimension se").grid(row=5, column=3)
    w20 = Entry(root, bd=5, textvariable=A20).grid(row=5, column=4)
    L21 = Label(root, text="Radius Worst").grid(row=6, column=3)
    w21 = Entry(root, bd=5, textvariable=A21).grid(row=6, column=4)
    L22 = Label(root, text="Texture Worst").grid(row=7, column=3)
    w22 = Entry(root, bd=5, textvariable=A22).grid(row=7, column=4)
    L23 = Label(root, text="Perimeter Worst").grid(row=8, column=3)
    w23 = Entry(root, bd=5, textvariable=A23).grid(row=8, column=4)
    L24 = Label(root, text="Area Worst").grid(row=9, column=3)
    w24 = Entry(root, bd=5, textvariable=A24).grid(row=9, column=4)
    L25 = Label(root, text="Smoothness Worst").grid(row=10, column=3)
    w25 = Entry(root, bd=5, textvariable=A25).grid(row=10, column=4)
    L26 = Label(root, text="Compactness Worst").grid(row=11, column=3)
    w26 = Entry(root, bd=5, textvariable=A26).grid(row=11, column=4)
    L27 = Label(root, text="Concavity Worst").grid(row=12, column=3)
    w27 = Entry(root, bd=5, textvariable=A27).grid(row=12, column=4)
    L28 = Label(root, text="Concave Points Worst").grid(row=13, column=3)
    w28 = Entry(root, bd=5, textvariable=A28).grid(row=13, column=4)
    L29 = Label(root, text="Symmetry Worst").grid(row=14, column=3)
    w29 = Entry(root, bd=5, textvariable=A29).grid(row=14, column=4)
    L30 = Label(root, text="Fractal Dimension Worst").grid(row=15, column=3)
    w30 = Entry(root, bd=5, textvariable=A30).grid(row=15, column=4)
    L30 = Label(root, text="").grid(row=16, column=3)
    B = tkinter.Button(root, text="     Output     ",
                       command=out).grid(row=19, column=4)
    L30 = Label(root, text="").grid(row=20, column=3)


# Prediction System
X_new = (A1.get(), A2.get(), A3.get(), A4.get(), A5.get(), A6.get(), A7.get(), A8.get(), A9.get(), A10.get(), A11.get(), A12.get(), A13.get(), A14.get(), A15.get(),
         A16.get(), A17.get(), A18.get(), A19.get(), A20.get(), A21.get(), A22.get(), A23.get(), A24.get(), A25.get(), A26.get(), A27.get(), A28.get(), A29.get(), A30.get())
input_data_as_numpy_array = np.asarray(X_new)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
Y_new = model.predict(input_data_reshaped)

# Output


def out():
    if(Y_new == 'B'):
        TkMessageBox.showinfo('Tissue Type', "Benign - Non-Cancerous")
    else:
        TkMessageBox.showinfo('Tissue Type', "Malignant - Cancerous")


fun()
root.mainloop()

# 'B' --> Benign --> Non-Cancerous
# 'M' --> Malignant --> Cancerous
