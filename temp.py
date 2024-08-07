import csv
import json

label_dict = {}
with open("hw4_data/office/train.csv", 'r') as file:
    csvreader = csv.reader(file)
    label = 0
    for i, row in enumerate(csvreader):
        #print(row[2])
        if i==0:
            continue
        if row[2] not in label_dict:
            label_dict[row[2]] = label
            label += 1
with open("sample.json", "w") as outfile:
    json.dump(label_dict, outfile, indent=4)
#for row in csvreader:
#print(row)
#print(csvreader)