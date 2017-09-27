import csv


def load_access():
    x = []
    y = []

    file = open("data/acesso_pagina.csv", "rb")
    csv_reader = csv.reader(file)

    csv_reader.next()
    for home, como_funciona, contato, comprou in csv_reader:
        x.append([int(home), int(como_funciona), int(contato)])
        y.append(int(comprou))

    return x, y
