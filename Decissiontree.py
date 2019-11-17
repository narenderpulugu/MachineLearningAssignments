import os
import pandas as pd
import numpy as np
import getopt
import sys
import math
import sklearn
import xml.etree.cElementTree as CET


def InformationGain(Entropy):
    InfoGain = IntialEntropy - Entropy
    return InfoGain


def Decission_Tree(data, root):
    bestattribute, entropy = BestAttribute(data)
    for attrvalues in data[bestattribute].unique():
        BufferData = data[data[bestattribute] == attrvalues]
        BufferData = BufferData.drop(columns = bestattribute)
        child_best_attribute, child_best_entropy = BestAttribute(BufferData)
        if child_best_attribute in leaflist:
            CET.SubElement(root, 'node', {'entropy': str(child_best_entropy), 'feature': bestattribute, 'value': attrvalues}).text = str(
                child_best_attribute)
            continue
        else:
            next_node = CET.SubElement(root, 'node', {'entropy': str(child_best_entropy), 'feature': bestattribute, 'value': attrvalues})
        Decission_Tree(BufferData, next_node)


def BestAttribute(data):
    attrEntropy = 0
    rootattribute = ""
    attrEntropyList = {}
    IntialEntropy = 0
    for Intial_value in np.array(data.Yattr.value_counts(normalize=True)):
        IntialEntropy = IntialEntropy - Intial_value * math.log(Intial_value, logbase)

    if IntialEntropy == 0:
        rootattribute = data.Yattr.unique()[0]
        return rootattribute, IntialEntropy

    for attr in data.columns[0:-1]:
        attrEntropy = 0
        BufferData = data[[attr, "Yattr"]]
        TotalInstances = len(BufferData.index)
        for attrvalues in data[attr].unique():
            AttrvalueData = BufferData[BufferData[attr] == attrvalues]
            totalattrinstances = len(AttrvalueData.index)
            for value in np.array(AttrvalueData.Yattr.value_counts(normalize=True)):
                attrEntropy = attrEntropy + (float(totalattrinstances) / TotalInstances) * (
                        - value * math.log(value, logbase))
        attrEntropyList[attr] = attrEntropy

    attrseries = pd.Series(attrEntropyList)
    attrgainSeries = -attrseries.subtract(IntialEntropy)
    rootattribute = attrgainSeries.idxmax()

    attrEntropyvalue = attrseries.get(key = rootattribute)

    # if all(value == 0 for value in attrEntropyList.values()):
    #     rootattribute = data.Yattr.unique()[0]
    #     attrEntropyvalue = 0

    return rootattribute, IntialEntropy


Data = pd.read_csv("/home/narender/GIT/MachineLearningAssignment/MachineLearningAssignments/car.csv",header= None, prefix= "Attr")
Data = Data.rename({Data.columns[-1]:"Yattr"}, axis=1)
attrlist = np.array(Data.columns[0:-1])
leaflist = Data["Yattr"].unique()
logbase = len(Data.Yattr.unique())
Ratio = (Data.Yattr.value_counts(normalize = True))
Ratio = np.array(Ratio)
IntialEntropy = 0
output_filename = "solution_for_xml_file.xml"

for ratio_value in Ratio:
    IntialEntropy = IntialEntropy - ratio_value* math.log(ratio_value,logbase)

attrEntropyList = {}
rootattribute = ""

root = CET.Element('tree', {'entropy': str(IntialEntropy)})
Decission_Tree(Data, root)
tree = CET.ElementTree(root)
tree.write(output_filename)


























