#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 11:24:46 2020

@author: nikhil
"""
import pandas as pd

data = pd.read_csv('dataset/train.csv')

smiles = data['SMILES'].tolist()
smells = data['SENTENCE'].tolist()

unique_smells = []
for smell in smells:
    l = smell.split(',')
    for s in l:
        if(s not in unique_smells):
            unique_smells.append(s)

unique_characters = []
for smile in smiles:
    for char in smile:
        if(char not in unique_characters):
            unique_characters.append(char)

def smile_to_code(smile):
    code = ''
    for char in smile:
        if(char in unique_characters):
            index = unique_characters.index(char) + 1
            if(index < 10):
                code += '0' + str(index) + ','
            else:
                code += str(index) + ','
        else:
            code += '99' + ','

    pad = 406 - len(code)*(2/3)
    code = '00,'*int(pad/2) + code[:-1]
    return code

def code_to_smile(code):
    smile = ''
    for char in code.split(','):
        if(char == '00'):
            continue
        if(char == '99'):
            smile += 'X'
        else:
            smile += unique_characters[int(char) - 1]
    return smile

dataset = ''

for smile,smell in zip(smiles,smells):
    l = smell.split(',')
    for s in l:
        dataset+= smile_to_code(smile) + ',' + str(unique_smells.index(s) + 1) + '\n'

data_test = pd.read_csv('dataset/test.csv')
smiles_test = data_test['SMILES'].tolist()

with open('dataset/dataset_test.csv','w') as file:
    for smile in smiles_test:
        file.write(smile_to_code(smile))
        file.write('\n')


with open('dataset/dataset.csv','w') as file:
    file.write(dataset)

with open('dataset/unique_smells.csv','w') as file:
    for smell in unique_smells:
        file.write(smell)
        file.write('\n')
    
with open('dataset/unique_characters.csv','w') as file:
    for char in unique_characters:
        file.write(char)
        file.write('\n')


