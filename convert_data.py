import os
import csv

dirname = os.path.dirname(__file__)
inpath = os.path.join(dirname, "training_data.txt")

with open(inpath) as infile:
    data = infile.readlines()

outpath = os.path.join(dirname, "training_data.csv")
with open(outpath, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    data = [row.strip("\n") for row in data]
    data = [row.split(" ") for row in data]
    data = [list(filter(lambda x : x != "", row)) for row in data]

    writer.writerow(["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"])
    writer.writerows(data)

