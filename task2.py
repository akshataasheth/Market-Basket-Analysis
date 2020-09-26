from pyspark import SparkContext, SparkConf
import sys
import json
from operator import add
import string
import csv
from itertools import combinations
import copy
import math
import time
from functools import reduce


start_time = time.time()

BUCKET_NUMBER = 1000000


def reformatItemset(itemset):
    i = 1
    result = ""
    for item in itemset:
        if len(item) == 1:
            #print(str(item)[1:-2])
            result += str("(" + str(item)[1:-2] + "),")
        elif len(item) == i:
            result += (str(item) + ",")
        else:
            result = result[:-1]
            result += "\n\n" + (str(item) + ",")
            i = len(item)
    return result[:-1]


def countFrequentItemset(data_baskets, candidate_pairs):
    temp_counter = {}
    for pairs in candidate_pairs:
        if set(pairs).issubset(set(data_baskets)):
            if pairs not in temp_counter.keys():
                temp_counter[pairs] = 1
            else:
                temp_counter[pairs] += 1
    yield [tuple((key, value)) for key, value in temp_counter.items()]


def getCandidateItems(frequentItems):
    candidateItems = list()
    if frequentItems is not None and len(frequentItems)>0:
        for index, x in enumerate(frequentItems[:-1]):
            for y in frequentItems[index+1:]:
                if(x[:-1]==y[:-1]):
                    new_combination = tuple(sorted(list(set(x).union(set(y)))))
                    candidateItems.append(new_combination)
                else:
                    break
        return candidateItems


def hashPair(combination):
    sum=0
    for x in list(combination):
        for chr in x:
            c = ord(chr)
            sum +=c
    return sum % BUCKET_NUMBER


def PCY(partition, support,fullBasketSize):
    partition = copy.deepcopy(list(partition))
    partitionSupport = support * (len(list(partition)))/(fullBasketSize)
    basketsChunk = list(partition)
    singletons = dict()
    i = 1
    bitVector = [0]*BUCKET_NUMBER       #Initialising bitVector to 0
    for basket in basketsChunk:
        for item in list(basket):
            if not item in singletons:
                singletons[item] = 1
            else:
                singletons[item] = singletons[item] + 1   
        for pair in combinations(list(basket),2):
            key = hashPair(pair)
            bitVector[key] = bitVector[key] + 1

    bitVector = list(map(lambda x: 1 if x>=partitionSupport else 0,bitVector))
    frequentSingletons = sorted(list(dict((filter(lambda x: x[1]>=partitionSupport, singletons.items()))).keys()))

    outputCandidates = {}
    i = 1
    outputCandidates[str(i)] = (tuple(item.split(",")) for item in frequentSingletons)
    candidateItems = frequentSingletons

    while None is not candidateItems and len(candidateItems)>0:
        i+=1
        candidateList = {}
        for basket in basketsChunk:
            basket = sorted(list(set(basket).intersection(set(frequentSingletons))))
            if len(basket)>=i:
                if i==2:
                    for pair in combinations(basket, 2):
                        if bitVector[hashPair(pair)]:
                            if pair not in candidateList.keys():
                                candidateList[pair] = 1
                            else:
                                candidateList[pair] += 1
                else:
                    for item in candidateItems:
                        if set(item).issubset(set(basket)):
                            if item not in candidateList.keys():
                                candidateList[item] = 1
                            else:
                                candidateList[item] += 1
        frequentItems = dict((filter(lambda x: x[1]>=partitionSupport, candidateList.items())))
        candidateItems = getCandidateItems(sorted(list(frequentItems.keys())),i)
        if len(frequentItems) == 0:  #check this
            break
        outputCandidates[str(i)] = list(frequentItems)
    return outputCandidates.values()


if __name__ == "__main__":
    #inputFile = 'user_business.csv'
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    inputFile = sys.argv[3]
    outputFile = sys.argv[4]
    conf = SparkConf().setAppName("Task2").setMaster("local[*]")
    sc = SparkContext(conf = conf)

    dataset = sc.textFile(inputFile).mapPartitions(lambda x : csv.reader(x))
    header = dataset.first()
    dataset = dataset.filter(lambda x: x != header)

    marketBaskets = dataset.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).filter(lambda x: len(x[1])>filter_threshold)
    marketBaskets = marketBaskets.map(lambda x: x[1])
    marketBaskets.persist()
    fullBasketSize = marketBaskets.count()

    #Phase 1 of SON Algorithm
    candidateItemset = marketBaskets.mapPartitions(lambda partition: PCY(partition, support, fullBasketSize)).flatMap(lambda pairs: pairs).distinct().sortBy(lambda pairs: (len(pairs), pairs)).collect()

    
    #phase 2 of SON Algorithm
    frequentItemset = marketBaskets.flatMap(lambda partition: countFrequentItemset(partition,candidateItemset)).flatMap(lambda pairs: pairs).reduceByKey(add).filter(lambda pair_count: pair_count[1] >= int(support)).map(lambda pair_count: pair_count[0]).sortBy(lambda pairs: (len(pairs), pairs)).collect()

    with open(outputFile, 'w+') as output_file:
        str_result = 'Candidates:\n' + reformatItemset(candidateItemset) + '\n\n' \
                     + 'Frequent Itemsets:\n' + reformatItemset(frequentItemset)
        output_file.write(str_result)
        output_file.close()

    print("--- Duration: %s seconds ---" % (time.time() - start_time))




        
