from sre_parse import Tokenizer
from pathlib import Path
import re
import numpy
from essential_generators import DocumentGenerator
import pandas as pd
import os
import math

from scipy import sparse
from scipy.sparse.linalg import svds

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

LEN_VOC = 0
def generate_texts():
    for i in range(1001):
        generator = DocumentGenerator()
        file_name = 'text' + str(i) + '.txt'
        path = 'texts/' + file_name
        string_to_write = generator.gen_sentence(50, 100)
        file = open(path, 'w')
        file.write(string_to_write)
        file.close()
        # os.remove(path)



def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    cleaned_text2 = [w.replace(",", "") for w in cleaned_text]
    cleaned_text3 = [w.replace(".", "") for w in cleaned_text2]
    return cleaned_text3


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        words = sorted(list(set(words)))
    return words

def generate_bow():
    allsentences = texts_to_array()
    global vocab
    vocab = tokenize(allsentences)
    # print("Word List for Document \n{0} \n".format(vocab))
    bag_arr = []
    LEN_VOC = len(vocab)
    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = [0]*LEN_VOC
        for w in words:
            for i,word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1

        bag_arr.append(bag_vector)


        #print("{0}\n{1}\n".format(words,numpy.array(bag_vector)))
    numpy_array = numpy.array(bag_arr)
    transpose = numpy_array.T
    global matrix_before_trans
    matrix_before_trans = numpy_array.tolist()
    transpose_list = transpose.tolist()
    return transpose_list, vocab

def texts_to_array():
    allsentences = []
    for i in range(1001):
        file_name = 'text' + str(i) + '.txt'
        path = 'texts/' + file_name
        sentence = Path(path).read_text()
        allsentences.append(sentence)
    return allsentences

def inverse_document_frequency():
    global document_matrix
    document_matrix, voc = generate_bow()
    LEN_VOC = len(vocab)
    counter_words = [0] * (LEN_VOC)
    for i in range(LEN_VOC):
        for j in range(1001):
            if document_matrix[i][j] != 0:
                counter_words[i] += 1
        if counter_words[i] != 0:
            counter_words[i] = math.log(1001/counter_words[i])

    for i in range(1001):
        for j in range(LEN_VOC):
            document_matrix[j][i] *= counter_words[j]

    return document_matrix

#compare 'texts/text1001.txt'
def input_to_bag_of_words():
    input_string= input(">>")
    LEN_VOC = len(vocab)
    words = word_extraction(input_string)
    bag_vector = [0] * LEN_VOC

    for w in words:
        for i, word in enumerate(vocab):
            if word == w:
                bag_vector[i] += 1

    return bag_vector

def cos(vector1, vector2):
    cos = 0
    for i in range(len(vector1)):
        cos += (vector1[i]*vector2[i])
    len1 = 0
    len2 = 0
    for i in range(len(vector2)):
        len1 += vector1[i]**2
        len2 += vector2[i]**2
    len1 = numpy.sqrt(len1)
    len2 = numpy.sqrt(len2)

    cos = cos/(len1*len2)
    return cos

def k_most_similar_texts(k):
    compare_document = []
    print(matrix_before_trans)
    input_bag_of_words = input_to_bag_of_words()
    for i in range(1001):
        similiar = cos(matrix_before_trans[i], input_bag_of_words)
        t = (similiar, i)
        compare_document.append(t)
    compare_document.sort(key = lambda x: x[0], reverse = True)

    print("the most similar texts:")

    for i in range(k):
        file_name = 'text' + str(compare_document[i][1]) + '.txt'
        path = 'texts/' + file_name
        print("{0} similiarity: {1}".format(path, compare_document[i][0]))

def normalize(v):
    norm=numpy.linalg.norm(v, ord=1)
    if norm==0:
        norm=numpy.finfo(v.dtype).eps
    return v/norm

def k_most_similar_texts_normalize(k):
    compare_document = []

    input_bag_of_words = input_to_bag_of_words()
    normalized_input_bag = normalize(input_bag_of_words)
    normalize_matrix()
    for i in range(1001):
        similiar = cos(normalized_matrix[i], normalized_input_bag)
        t = (similiar, i)
        compare_document.append(t)
    compare_document.sort(key = lambda x: x[0], reverse = True)

    print("the most similar texts:")

    for i in range(k):
        file_name = 'text' + str(compare_document[i][1]) + '.txt'
        path = 'texts/' + file_name
        print("{0} similiarity: {1}".format(path, compare_document[i][0]))

def normalize_matrix():
    global normalized_matrix
    normalized_matrix = []
    for i in range(1001):
        normalized_matrix.append(normalize(matrix_before_trans[i]))


def svd_matrix(k_svd):
    U, S, VT = sparse.linalg.svds(normalized_matrix, k_svd)
    svd_matrix = U @ numpy.diag(S) @ VT
    return svd_matrix

def k_most_similar_texts_noise_removal(k, k_svd):
    compare_document = []

    input_bag_of_words = input_to_bag_of_words()
    normalized_input_bag = normalize(input_bag_of_words)
    normalize_matrix()
    svd = svd_matrix(k_svd)
    for i in range(1001):
        similiar = cos(svd[i], normalized_input_bag)
        t = (similiar, i)
        compare_document.append(t)
    compare_document.sort(key=lambda x: x[0], reverse=True)

    print("the most similar texts:")

    for i in range(k):
        file_name = 'text' + str(compare_document[i][1]) + '.txt'
        path = 'texts/' + file_name
        print("{0} similiarity: {1}".format(path, compare_document[i][0]))

def k_most_similar_texts_noise_removal_test(k):
    for pow in [2,3,4,5,6,7,8,9]:
        print(f"{bcolors.HEADER} k = {2**pow} {bcolors.ENDC}")
        k_most_similar_texts_noise_removal(k, 2**pow)


def print_tmp():
    np_array = generate_bow()
    for i in range(5):
        for j in range(5):
            print(np_array.p)


if __name__ == "__main__":


    # generate_texts()
    generate_bow()
    # print_tmp()
    # print(texts_to_array())
    # print(sentence_text0)
    # print(tokenize(sentence))
    inverse_document_frequency()
    # k_most_similar_texts(2)
    # k_most_similar_texts_normalize(2)
    k_most_similar_texts_noise_removal_test(2)

    #

