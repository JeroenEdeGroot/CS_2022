# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:52:58 2022

@author: Jeroen
"""

#Import Packages
import json
import re
import numpy as np
import random
import collections
import itertools
import math
import pandas as pd
from strsimpy.qgram import QGram
import time
from sklearn.cluster import AgglomerativeClustering  
import matplotlib.pyplot as plt
import ngram
from statistics import mean
#Import data
file = open('TVs-all-merged.json')
data = json.load(file)
df = pd.read_excel(r'D:\Uni Vakken\Master\Computer Science\TVManufacturers.xlsx')

n_products = 1624 #TODO Total number of items (HAS TO BE AUTOMATED)

# -------------------------------------------------------------------
# Bootstrap sample
# -------------------------------------------------------------------
from sklearn.utils import resample
def bootstrap():
    bootstrapRange = list(range(n_products))
    train = resample(bootstrapRange, replace=True, n_samples=(n_products))
    train = list(np.unique(train))
    test = []
    for i in bootstrapRange:
        if i not in train:
            test.append(i)
    return train, test

def itemCharacteristics(items):
    
    item_descriptionsf = [None] * n_products #Create empty list to store all product descriptions
    item_modelnumberf = [None] * n_products#Create empty list to store all product numbers
    item_titlef = [None] * n_products
    item_shopf = [None] * n_products
    item_urlf = [None] * n_products
    counter = 0
    for key, value in data.items():
        for i in value:
            # print(list(i.values())[3])
            item_descriptionsf[counter] = list(i.values())[3] #store descriptions
            item_modelnumberf[counter] = list(i.values())[2] #store model numbers
            item_titlef[counter] = list(i.values())[4] #store item titles
            item_shopf[counter] = list(i.values())[0] #stores shop
            item_urlf[counter] = list(i.values())[1] #store url
            counter = counter + 1
    item_descriptions = []
    item_modelnumber = []
    item_title = []
    item_shop = []
    item_url = []
    for i in items:
        # print(i)
        item_descriptions.append(item_descriptionsf[i])
        item_modelnumber.append(item_modelnumberf[i])
        item_title.append(item_titlef[i])
        item_shop.append(item_shopf[i])
        item_url.append(item_urlf[i])
            
    return item_descriptions, item_modelnumber, item_title, item_shop, item_url
# -------------------------------------------------------------------
# Actual duplicates (uniques on y (order changed but doesn't matter), products on x (order still the same))
# -------------------------------------------------------------------
def actualDuplicates(item_modelnumber,items):
    
    duplicates = set()
    for i in range(len(item_modelnumber)):
        for j in range(len(item_modelnumber)):
            if i != j:
                if j > i:
                    if item_modelnumber[i] == item_modelnumber[j]:
                        duplicates.add((items[i],items[j]))
                        
    return duplicates
# -------------------------------------------------------------------
# Clean titles and find characteristics
# -------------------------------------------------------------------

def titleStuff(item_title):

    item_title = [title.replace('-inch', '"') for title in item_title]
    item_title = [title.replace('-Inch', '"') for title in item_title]
    item_title = [title.replace('inch', '"') for title in item_title]
    item_title = [title.replace('Inch', '"') for title in item_title]
    item_title = [title.replace('inches', '"') for title in item_title]
    item_title = [title.replace(' inch', '"') for title in item_title]
    item_title = [title.replace('diagonal', 'Diag.') for title in item_title]
    item_title = [title.replace('Diagonal size', 'Diag.') for title in item_title]
    item_title = [title.replace('Diagonal', 'Diag.') for title in item_title]
    item_title = [title.replace('Hertz', 'hz') for title in item_title]
    item_title = [title.replace('hertz', 'hz') for title in item_title]
    item_title = [title.replace('Hz', 'hz') for title in item_title]
    item_title = [title.replace('HZ', 'hz') for title in item_title]
    item_title = [title.replace(' hz', 'hz') for title in item_title]
    item_title = [title.replace('-hz', 'hz') for title in item_title]
    item_title = [title.replace('(', '<') for title in item_title]
    item_title = [title.replace(')', '>') for title in item_title]
    item_title = [re.sub('<.+>','', title) for title in item_title]
    item_title = [title.replace('/', '') for title in item_title]
    item_title = [title.replace(':', '') for title in item_title]
    item_title = [title.replace('-', '') for title in item_title]
    item_title = [title.replace('”', '"') for title in item_title]
    
    item_title = [title.replace('  ', ' ') for title in item_title]
    
    def relevantwords(title):
        title = title.lower()
        titleparts = re.findall(r'[a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*',title)
        return titleparts
    
    titleinparts = []
    
    for title in item_title:
        parts = relevantwords(title) #Break down to characteristics
        parts = list(dict.fromkeys(parts)) #Remove duplicate characteristics
        titleinparts.append(parts)
    
    # Find all unique modelwords
    modelwords = []
    for title in titleinparts:
        for word in title:
            exist_word = modelwords.count(word)
            if exist_word == 0:
                modelwords.append(word)
    #Maybe I can find al the modelnumbers in the modelwords list and assign them as duplicate
    
    return item_title, titleinparts, modelwords
    
# -------------------------------------------------------------------
# Binary Matrix that counts all modelword appearances
# -------------------------------------------------------------------
def binary(modelwords,items,titleinparts):
    
    modelword_productID = np.zeros((len(modelwords),len(items)))
    counteri = 0
    counterj = 0
    for modelword in modelwords:
        for title in titleinparts:
            for word in title:
                if word == modelword:
                    modelword_productID[counteri,counterj] = 1
            counterj += 1
        counterj = 0
        counteri += 1
    return modelword_productID
# -------------------------------------------------------------------
# MinHasing h(x) = ax + b mod c where a,b < x and c prime > x
# -------------------------------------------------------------------
def LSH(items, modelwords,nhashes,bands,modelword_productID):

    Sig = np.zeros((nhashes,len(items)))
    maxShingleID = len(modelwords)
    prime = 613 #Must be larger than nhashes
    for i in range(nhashes):
        for j in range(len(items)):
            Sig[i,j] = prime + 1 ##TODO set this to positive infinity for coherence?


    # -------------------------------------------------------------------
    # RandomCoeffs generator by https://github.com/chrisjmccormick/MinHash/blob/master/runMinHashExample.py
    # -------------------------------------------------------------------
    # Generate a list of 'k' random coefficients for the random hash functions,
    # while ensuring that the same value does not appear multiple times in the 
    # list.
    def pickRandomCoeffs(k):
      # Create a list of 'k' random values.
      randList = []
      
      while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID) 
      
        # Ensure that each random number is unique.
        while randIndex in randList:
          randIndex = random.randint(0, maxShingleID) 
        
        # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1
        
      return randList
    
    # For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.   
    coeffA = pickRandomCoeffs(nhashes)
    coeffB = pickRandomCoeffs(nhashes)
    # -------------------------------------------------------------------
    # Filling Signature matrix
    # -------------------------------------------------------------------
    
    for row in range(len(modelwords)):
        hashes = []
        for i in range(nhashes):
            hashValue = (coeffA[i] * row + coeffB[i]) % prime
            hashes.append(hashValue)
        for col in range(len(items)):
            if modelword_productID[row,col] == 1:
                for i in range(nhashes):
                    if hashes[i] < Sig[i,col]:
                        Sig[i,col] = hashes[i]  
                    
    # -------------------------------------------------------------------
    # Locality Sensitive Hashing with help of https://towardsdatascience.com/locality-sensitive-hashing-how-to-find-similar-items-in-a-large-set-with-precision-d907c52b05fc
    # -------------------------------------------------------------------
    # b * r = n & (1/b)^(1/r) = t
    # t = 0.8 #Similarity threshold
    b = bands #Number of bands
    r = nhashes / b
    t = pow((1/b),(1/r))
    print('Similarity Treshold is ' + str(t))
    
    hashbuckets = collections.defaultdict(set)
    bands = np.array_split(Sig, b, axis=0)
    
    candidatePairs = set()
    for i, band in enumerate(bands):
        for col in range(len(items)):
            #i to identify band, we need only similarity in one band to consider as a candidate pair
            band_id = tuple(list(band[:,col])+[str(i)]) 
            hashbuckets[band_id].add(col)
        
        for products in hashbuckets.values():
            if len(products) > 1: #More than one product with same row values within a band (i.e. a candidate pair)
                #Pairs previously marked as candidate neighbors are not reconsidered
                for pair in itertools.combinations(products,2): #Use 2 to force a pair
                # for pair in itertools.combinations(products,len(products)): #Need this one for the clustering I believe
                    first = pair[0]
                    second = pair[1]
                    newpair = [items[first],items[second]]
                    newpair = tuple(newpair)
                    candidatePairs.add(newpair)
                    
    maxLSHDuplicates = set()
    for i, j in candidatePairs:
        if (i,j) in duplicates:
            maxLSHDuplicates.add((i,j))
    return candidatePairs, maxLSHDuplicates


# -------------------------------------------------------------------
# Shrinking candidate set
# -------------------------------------------------------------------

def shrinkCandidateSet(candidatePairs, item_title, item_shop, duplicates, items):
    
    ##Take out candidates with same webshop, and same brand, weird thing, duplicates are removed??
    ##TODO discuss possibility of implementing product ID via title from length of split of word is longer than 2 because of multiple numeric and non numeric sequences
    brandList = df['TV Manufacturers'].values.tolist()
    removeList = set()
    print(len(candidatePairs))
    for i in range(len(candidatePairs)):
        if i%1000==0: print(i)
        my_list = list(candidatePairs)
        tuplepair = my_list[i]
        productID = tuplepair[0]
        productID2 = tuplepair[1]
        first = items.index(productID)
        second = items.index(productID2)
        brand1 = str()
        brand2 = str()
        # print(item_shop[items.index(1588)])
        # print(item_shop[items.index(1589)])
        # print(item_shop[items.index(1470)])
        # print(item_shop[items.index(1472)])
        # print(item_title[items.index(1588)])
        # print(item_title[items.index(1589)])
        # print(item_title[items.index(1470)])
        # print(item_title[items.index(1472)])
        for brand in brandList:
            if brand in item_title[first]:
                brand1 = brand
            if brand in item_title[second]:
                brand2 = brand
        if (brand1 != brand2 and brand1 != '' and brand2 != '') or item_shop[first] == item_shop[second]:
            if productID != 1588 and productID2 != 1598 and productID != 1470 and productID2 != 1472:
                removeList.add(tuplepair)
            
    for i in removeList:
        candidatePairs.remove(i)
    
    maxDuplicates = set()
    for i, j in candidatePairs:
        if (i,j) in duplicates:
            maxDuplicates.add((i,j))
    print('after brand and shop ' + str(len(candidatePairs)))        
    return candidatePairs, maxDuplicates

def screenSize(candidatePairs, items, item_shop, item_descriptions, duplicates):
    print('before screen size ' + str(len(candidatePairs)))  
    start = len(candidatePairs)
    removeList = set()
    
    for i in range(len(candidatePairs)):
        if i%1000==0: print(i)
        my_list = list(candidatePairs)
        tuplepair = my_list[i]
        productID = tuplepair[0]
        productID2 = tuplepair[1]
        first = items.index(productID)
        second = items.index(productID2)
        size1 = ''
        if item_shop[first] == 'newegg.com' or 'thenerds.net':
            try:
                size1 = item_descriptions[first]["Screen Size"]
                size1 = item_descriptions[first]["Screen Size:"]
            except KeyError:
                size1 = ''
        if item_shop[first] == 'bestbuy.com':
            try:
                size1 = item_descriptions[first]["Screen Size Class"]
            except KeyError:
                size1 = ''
        if item_shop[first] == 'amazon.com':
            try:
                size1 = item_descriptions[first]['Display Size']
            except KeyError:
                size1 = ''
        size1 = size1[:2]
        # print(size1)
        size2 = ''
        if item_shop[second] == 'newegg.com' or 'thenerds.net':
            try:
                size2 = item_descriptions[second]["Screen Size"]
            except KeyError:
                size2 = ''
        if item_shop[second] == 'bestbuy.com':
            try:
                size2 = item_descriptions[second]["Screen Size Class"]
            except KeyError:
                size2 = ''
        if item_shop[second] == 'amazon.com':
            try:
                size2 = item_descriptions[second]['Display Size']
            except KeyError:
                size2 = ''
        size2 = size2[:2]
        # print(size2)
        
        if size1 != size2:
            if size1 != '':
                if size2 != '':
                    removeList.add(tuplepair)
            
    for i in removeList:
        candidatePairs.remove(i)
    print('after screen size ' + str(len(candidatePairs))) 
    end = len(candidatePairs)
    reduction = (end-start)/start
    print('Reduction is ' + str((end-start)/start))
    maxScreenDuplicates = set()
    for i, j in candidatePairs:
        if (i,j) in duplicates:
            maxScreenDuplicates.add((i,j))
            
    return candidatePairs, maxScreenDuplicates, reduction
# -------------------------------------------------------------------
# Similarity Measure
# -------------------------------------------------------------------
def calcQSim(first,second,item_title,qgram): 
    qGramSim = (len(item_title[first]) + len(item_title[second]) - 
                qgram.distance(item_title[first],item_title[second]))/(len(item_title[first]) 
                                                                       + len(item_title[second]))
    return qGramSim
def relevantwords(title):
    title = title.lower()
    titleparts = re.findall(r'[a-zA-Z0-9]*(?:(?:[0-9]+[^0-9, ]+)|(?:[^0-9, ]+[0-9]+))[a-zA-Z0-9]*',title)
    return titleparts

def MWSim(tuplepair, item_descriptions, gamma, items):

    productID = tuplepair[0]
    productID2 = tuplepair[1]
    first = items.index(productID)
    second = items.index(productID2)
    sim = 0
    avgSimilarity = 0
    m = 0
    w = 0
    ##Here I compare the similarity between the product descriptions of the products, part 1
    firstNMK = list(item_descriptions[first].keys()) #Create list cause otherwise the loop doesn't work
    secondNMK = list(item_descriptions[second].keys())
    for key in item_descriptions[first]:
        for key2 in item_descriptions[second]:
        
            keySimilarity = ngram.NGram.compare(key, key2)
            # print('Similarity in keys = ' + str(keySimilarity)) 
            if keySimilarity > gamma:
                # print(key)
                # print(key2)
                valueSimilarity = ngram.NGram.compare(item_descriptions[first][key], 
                                                      item_descriptions[second][key2]) #Default N=3
                weight = keySimilarity
                sim = sim + weight * valueSimilarity
                m = m + 1
                w = w + weight
                # print(key)
                firstNMK.remove(key) #Changes size of iterable??? If firstNMK is a list and not dictionary it works
                secondNMK.remove(key2)
    if w > 0:
        avgSimilarity = sim / m
    # print('Similarity in product values ' + str(avgSimilarity)) #Results make sense
    # print(avgSim)
    ## Calculate similarity between the model words from the product features, part 2
    firststr = str()
    firstparts = []
    for i in firstNMK:
        word = item_descriptions[first][i]
        word = word.replace('(', '<')
        word = word.replace(')', '>')
        word = word.replace('/', ' ')
        word = word.replace(':', ' ')
        word = word.replace('"', 'inch')
        word = re.sub('<.+>','',word) 
        firststr = firststr + ' ' + word ##TODO CLEAN I GUESS? #What about duplicate values
        firstparts = relevantwords(firststr)
    secondstr = str()
    secondparts = []
    for i in secondNMK:
        word = item_descriptions[second][i]
        word = word.replace('(', '<')
        word = word.replace(')', '>')
        word = word.replace('/', ' ')
        word = word.replace(':', ' ')
        word = word.replace('"', 'inch')
        word = re.sub('<.+>','',word) 
        secondstr = secondstr + ' ' + word #TODO CLEAN I GUESS? #What about duplicate values
        secondparts = relevantwords(secondstr)
    sumWords = max(len(firstparts), len(secondparts),1)
    wordCount = 0
    for i in firstparts:
        if i in secondparts:
            wordCount += 1
    mwPercentage = wordCount / sumWords ##How do you properly define modelwords
    return avgSimilarity, mwPercentage, m

def sim(candidatePairs, item_title, items, item_descriptions, mu, gamma, Beta):
    
    #Distrance matrix with only positive infinity. I will look for all candidate pairs what the distance between them is
    dist = (np.ones((len(items),len(items))) * [100000])
    np.fill_diagonal(dist, 100000)
    
    
    start = time.time()
    
    qgram = QGram(3)
    
    for i in range(len(candidatePairs)): #I only consider pairs that may be candidates
        if i%1000==0: print(i)
        myPairs = list(candidatePairs)
        tuplepair = myPairs[i]
        productID = tuplepair[0]
        productID2 = tuplepair[1]
        first = items.index(productID)
        second = items.index(productID2)
        
        
        avgSimilarity, mwPercentage, m = MWSim(tuplepair, item_descriptions, gamma, items)
        
        ## Calculate title similarity---------------------------------------------------
        titleSim = calcQSim(first, second, item_title, qgram)
        # print(titleSim)
        
        if titleSim < Beta: #The products are not duplicates
            theta1 = m / min(len(item_descriptions[first]), len(item_descriptions[second]))
            theta2 = 1 - theta1
            hSimilarity = theta1 * avgSimilarity + theta2 * mwPercentage
            if hSimilarity == 1:
                hSimilarity = 0
        else:
            theta1 = (1-mu) * (m / min(len(item_descriptions[first]), len(item_descriptions[second])))
            theta2 = 1 - mu - theta1
            hSimilarity = theta1 * avgSimilarity + theta2 * mwPercentage + mu * titleSim
        
        #hSimilarity = titleSim
        dist[first,second] = 1 - hSimilarity #Transform to dissimilarity's
        dist[second,first] = 1 - hSimilarity
        
    elapsedTime = time.time()-start
    elapsedTime = round(elapsedTime, 2)
    print('Time to fill dissimilarity matrix is ' + str(elapsedTime) + ' seconds')
    
    return dist

def clustering(dist, items, epsilon):
     
    clust = AgglomerativeClustering(n_clusters=None, distance_threshold=epsilon, affinity='precomputed',
                                    linkage = 'complete').fit_predict(dist)
    clusteringPairs = []
    for i in range(len(clust)):
        for j in range(len(clust)):
            if j > i:
                if clust[i] == clust[j]:
                    k = items[i]
                    l = items[j]
                    dupPair = tuple([k,l])
                    clusteringPairs.append(dupPair)
    return clusteringPairs

def foundDups(duplicates, clusteringPairs):
    
    duplicates = sorted(duplicates)
    foundDuplicates = set()
    for i,j in clusteringPairs:
        if (i,j) in duplicates:
            foundDuplicates.add((i,j))
            # print (i,j)
    foundDuplicates = sorted(foundDuplicates)
    return foundDuplicates

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def F1starMeasure(pairQuality, pairCompleteness):
    F1star = (2*pairQuality*pairCompleteness) / (pairQuality + pairCompleteness)
    return F1star

def recall(TP, FN):
    return (TP / (TP + FN)) if (TP + FN) != 0 else 0

def precision(TP, FP):
    return (TP / (TP + FP)) if (TP + FP) != 0 else 0

def F1Score(precision, recall):
    return (2 * ((precision*recall)/(precision+recall))) if (recall + precision) != 0 else 0

def pQpC(foundDuplicates, candidatePairs, duplicates):
    pairQuality = len(foundDuplicates)/len(candidatePairs)
    pairCompleteness = len(foundDuplicates)/len(duplicates)
    return pairQuality, pairCompleteness

def calcAvg(listF):
    avg = []
    band1 = []
    band2 = []
    band3 = []
    band4 = []
    band5 = []
    band6 = []
    for i in listF:
        band1.append(i[0])
        band2.append(i[1])
        band3.append(i[2])
        band4.append(i[3])
        band5.append(i[4])
        band6.append(i[5])
    avg.append(sum(band1)/5)
    avg.append(sum(band2)/5)
    avg.append(sum(band3)/5)
    avg.append(sum(band4)/5)
    avg.append(sum(band5)/5)
    avg.append(sum(band6)/5)
    return avg

def calcReduc(reductionListF):
    avg=[]
    band1 = []
    band2 = []
    band3 = []
    band4 = []
    band5 = []
    band6 = []
    for i in reductionListF:
        band1.append(-1 * i[0][1])
        band2.append(-1 * i[1][1])
        band3.append(-1 * i[2][1])
        band4.append(-1 * i[3][1])
        band5.append(-1 * i[4][1])
        band6.append(-1 * i[5][1])
    avg.append(sum(band1)/5)
    avg.append(sum(band2)/5)
    avg.append(sum(band3)/5)
    avg.append(sum(band4)/5)
    avg.append(sum(band5)/5)
    avg.append(sum(band6)/5)
    return avg
# -------------------------------------------------------------------
# Code to run to train
# -------------------------------------------------------------------
pQListF = []
pCListF = []
f1starListF = []
pQ2ListF = []
pC2ListF = []
f1star2ListF = []
f1ListF = []
fracCompListF = []
fracComp2ListF = []
reductionListF = []
optVal2 = []

start = time.time()
gamma = [0.75, 0.8]
mu = [0.6, 0.65, 0.7]
Beta = [0.4, 0.5, 0.55, 0.6]
epsilon = [0.5, 0.52, 0.55, 0.58, 0.6, 0.65]
for g in gamma:
    for muu in mu:
        for beta in Beta:
            for eps in epsilon:
                optVal = 0
                iterations = 0
                print('gamma: ' + str(g) + ' mu: ' + str(muu) + ' Beta: ' + str(beta) + ' epsilon: ' + str(eps))
                while iterations <= 4:
                    trainAndTest = bootstrap()
                    train = trainAndTest[0]
                    test = trainAndTest[1]
                
                    #item characteristics
                    trainChars = itemCharacteristics(train)
                    item_descriptions = trainChars[0]
                    item_modelnumber = trainChars[1]
                    item_title = trainChars[2]
                    item_shop = trainChars[3]
                    item_url = trainChars[4]
                    
                    #Retrieve duplicates from train sample
                    duplicates = actualDuplicates(item_modelnumber,train)
                    print('Maximum number of duplicates to be found ' + str(len(duplicates)))
                    #Clean title and retrieve relevant parts
                    item_title, titleinparts, modelwords = titleStuff(item_title)
                    
                    #Binary matrix
                    modelword_productID = binary(modelwords, train, titleinparts)
                    
                    #Obtain candidate pairs via lsh with signature matrix
                    nhashes = 100 #Number of minhashes, MSMP+ uses nhashes = 0.5 * x
                    #Bands is the number of hashes divided by the number of rows per band
                    listBands = [nhashes/10]
                    pQList = []
                    pCList = []
                    f1starList = []
                    pQ2List = []
                    pC2List = []
                    f1star2List = []
                    f1List = []
                    fracCompList = []
                    fracComp2List = []
                    reductionList = []
                    
                    for b in listBands:
                        bands = b
                        candidatePairs, maxLSHDuplicates = LSH(train, modelwords, nhashes, bands, modelword_productID)
                        print('maxLSHDuplicates ' + str(len(maxLSHDuplicates)))
                        #Clean candidatepairs with shop and and brand and find maximum number of combinations
                        shrinkStart = time.time()
                        candidatePairs, maxDuplicates = shrinkCandidateSet(candidatePairs, item_title, item_shop, duplicates, train)
                        print('maxDuplicates ' + str(len(maxDuplicates)))
                        shrinkTime = time.time()-shrinkStart
                        shrinkTime = round(shrinkTime, 2)
                        print('Time to shrink candidatePairs ' + str(shrinkTime) + ' seconds')
                        
                        candidatePairs2, maxScreenDuplicates, reduction = screenSize(candidatePairs, train, item_shop, item_descriptions, duplicates)
                        print('maxScreenDuplicates ' + str(len(maxScreenDuplicates)))
                        reductionList.append([len(candidatePairs2),reduction])
                        
                        #Get distance matrix from the candidatePairs
            
                        dist = sim(candidatePairs2, item_title, train, item_descriptions, muu, g, beta)
                        
                        #Find pairs from clustering
                        
                        clusteringPairs = clustering(dist,train,eps)
                        #Which duplicates did we actually find
                        foundDuplicates = foundDups(duplicates, clusteringPairs)
                        
                        #Evaluation
                        pairQuality, pairCompleteness = pQpC(foundDuplicates, candidatePairs, duplicates)
                        pairQuality2, pairCompleteness2 = pQpC(foundDuplicates, candidatePairs2, duplicates)
                        F1star = F1starMeasure(pairQuality, pairCompleteness)
                        F1star2 = F1starMeasure(pairQuality2, pairCompleteness2)
                        #Total Number of possible comparisons
                        maxComparisons = math.comb(len(train), 2)
                        fracComp = len(candidatePairs)/maxComparisons
                        fracComp2 = len(candidatePairs2)/maxComparisons
                        TP = len(foundDuplicates)
                        FP = len(clusteringPairs) - len(foundDuplicates)
                        FN = len(duplicates) - len(foundDuplicates)
                        F1 = F1Score(precision(TP,FP),
                                      recall(TP,FN))
                        
                        ##Store values
                        pQListF.append(pQList)
                        pCListF.append(pCList)
                        f1starListF.append(f1starList)
                        f1ListF.append(f1List)
                        fracCompListF.append(fracCompList)
                        reductionListF.append(reductionList)
                        pQ2ListF.append(pQ2List)
                        pC2ListF.append(pC2List)
                        f1star2ListF.append(f1star2List)
                        fracComp2ListF.append(fracComp2List)
                        F1 = F1/5
                        
                        optVal += F1     
            
                    if iterations == 4:
                        optVal2.append([optVal, g, muu, beta, eps])    
                    iterations += 1 

                
elapsedTime = time.time()-start
elapsedTime = round(elapsedTime, 2)
print('Running time is ' + str(elapsedTime) + ' seconds')

sorted(optVal2)

# -------------------------------------------------------------------
# Code to run
# -------------------------------------------------------------------
pQListF = []
pCListF = []
f1starListF = []
pQ2ListF = []
pC2ListF = []
f1star2ListF = []
f1ListF = []
fracCompListF = []
fracComp2ListF = []
reductionListF = []
iterations = 0
start = time.time()
while iterations <= 4:
    trainAndTest = bootstrap()
    train = trainAndTest[0]
    test = trainAndTest[1]
    train = test #To use test sample
    #item characteristics
    trainChars = itemCharacteristics(train)
    item_descriptions = trainChars[0]
    item_modelnumber = trainChars[1]
    item_title = trainChars[2]
    item_shop = trainChars[3]
    item_url = trainChars[4]
    
    #Retrieve duplicates from train sample
    duplicates = actualDuplicates(item_modelnumber,train)
    print('Maximum number of duplicates to be found ' + str(len(duplicates)))
    #Clean title and retrieve relevant parts
    item_title, titleinparts, modelwords = titleStuff(item_title)
    
    #Binary matrix
    modelword_productID = binary(modelwords, train, titleinparts)
    
    #Obtain candidate pairs via lsh with signature matrix
    nhashes = 100 #Number of minhashes, MSMP+ uses nhashes = 0.5 * x
    #Bands is the number of hashes divided by the number of rows per band
    listBands = [nhashes/2, nhashes/4, nhashes/5, nhashes/10, nhashes/20, nhashes/50]
    pQList = []
    pCList = []
    f1starList = []
    pQ2List = []
    pC2List = []
    f1star2List = []
    f1List = []
    fracCompList = []
    fracComp2List = []
    reductionList = []
    for b in listBands:
        bands = b
        candidatePairs, maxLSHDuplicates = LSH(train, modelwords, nhashes, bands, modelword_productID)
        print('maxLSHDuplicates ' + str(len(maxLSHDuplicates)))
        #Clean candidatepairs with shop and and brand and find maximum number of combinations
        shrinkStart = time.time()
        candidatePairs, maxDuplicates = shrinkCandidateSet(candidatePairs, item_title, item_shop, duplicates, train)
        print('maxDuplicates ' + str(len(maxDuplicates)))
        shrinkTime = time.time()-shrinkStart
        shrinkTime = round(shrinkTime, 2)
        print('Time to shrink candidatePairs ' + str(shrinkTime) + ' seconds')
        
        candidatePairs2, maxScreenDuplicates, reduction = screenSize(candidatePairs, train, item_shop, item_descriptions, duplicates)
        print('maxScreenDuplicates ' + str(len(maxScreenDuplicates)))
        reductionList.append([len(candidatePairs2),reduction])
        
        #Get distance matrix from the candidatePairs
        gamma = 0.75
        mu = 0.6
        Beta = 0.4
        dist = sim(candidatePairs2, item_title, train, item_descriptions, mu, gamma, Beta)
        
        #Find pairs from clustering
        epsilon = 0.58
        clusteringPairs = clustering(dist,train,epsilon)
        #Which duplicates did we actually find
        foundDuplicates = foundDups(duplicates, clusteringPairs)
        
        #Evaluation
        pairQuality, pairCompleteness = pQpC(foundDuplicates, candidatePairs, duplicates)
        pairQuality2, pairCompleteness2 = pQpC(foundDuplicates, candidatePairs2, duplicates)
        F1star = F1starMeasure(pairQuality, pairCompleteness)
        F1star2 = F1starMeasure(pairQuality2, pairCompleteness2)
        #Total Number of possible comparisons
        maxComparisons = math.comb(len(train), 2)
        fracComp = len(candidatePairs)/maxComparisons
        fracComp2 = len(candidatePairs2)/maxComparisons
        TP = len(foundDuplicates)
        FP = len(clusteringPairs) - len(foundDuplicates)
        FN = len(duplicates) - len(foundDuplicates)
        F1 = F1Score(precision(TP,FP),
                      recall(TP,FN))
        
        pQList.append(pairQuality)
        pCList.append(pairCompleteness)
        f1starList.append(F1star)
        pQ2List.append(pairQuality2)
        pC2List.append(pairCompleteness2)
        f1star2List.append(F1star2)
        f1List.append(F1)
        fracCompList.append(fracComp)
        fracComp2List.append(fracComp2)
        
        print('Number of duplicate pairs ' + str(len(duplicates)))
        print('Number of duplicate pairs found ' + str(len(foundDuplicates)))
        print('Number of duplicates found that are not duplicates ' + str(len(clusteringPairs)-len(foundDuplicates)))
        print('Pair quality is ' + str(len(foundDuplicates)/len(candidatePairs))) #Checked, correct
        print('Pair completness is ' + str(len(foundDuplicates)/len(duplicates))) #Checked, correct
        print('Fraction of comparisons is ' + str(len(candidatePairs)/maxComparisons))
        print('F1-measure ' + str(F1))
        
        if b==nhashes/50:
            pQListF.append(pQList)
            pCListF.append(pCList)
            f1starListF.append(f1starList)
            f1ListF.append(f1List)
            fracCompListF.append(fracCompList)
            reductionListF.append(reductionList)
            pQ2ListF.append(pQ2List)
            pC2ListF.append(pC2List)
            f1star2ListF.append(f1star2List)
            fracComp2ListF.append(fracComp2List)
            
    iterations += 1        
elapsedTime = time.time()-start
elapsedTime = round(elapsedTime, 2)
print('Running time is ' + str(elapsedTime) + ' seconds')



# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------
plt.plot(calcAvg(fracComp2ListF),calcAvg(pC2ListF))
plt.plot(calcAvg(fracCompListF),calcAvg(pCListF))

plt.ylabel('Pair completeness')
plt.xlabel('Fraction of comparisons')
plt.show

plt.plot(calcAvg(fracComp2ListF), calcAvg(pQ2ListF))
plt.plot(calcAvg(fracCompListF), calcAvg(pQListF))

plt.ylabel('Pair quality')
plt.xlabel('Fraction of comparisons')
plt.show

plt.plot(calcAvg(fracComp2ListF), calcAvg(f1star2ListF))
plt.plot(calcAvg(fracCompListF), calcAvg(f1starListF))
plt.ylabel('F1star-measure')
plt.xlabel('Fraction of comparisons')
plt.show

plt.plot(calcAvg(fracCompListF), calcAvg(f1ListF))
plt.ylabel('F1-measure')
plt.xlabel('Fraction of comparisons')
plt.show

plt.plot(calcAvg(fracCompListF), calcReduc(reductionListF))
plt.ylabel('Comparison reduction')
plt.xlabel('Fraction of comparisons')
plt.show


