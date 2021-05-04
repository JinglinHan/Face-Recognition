import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from modules import picProcessor, nnBaseModel


if __name__ == "__main__":
    ID_name = {'01': '陈立江', 'PT2000005': '张林秀',
               'PT2000006': '南佳豪', 'PT2000012': '姜雨薇',
               'PT2000054': '熊敏捷', 'PT2000437': '李聪昊',
               'P20000188': '于冲', 'SY1906903': '倪京成',
               'SY2002105': '蔡凌翰', 'SY2002107': '贺嘉诚',
               'SY2002110': '刘玉浩', 'SY2002112': '任洁',
               'SY2002114': '孙爱芃', 'SY2002406': '王胥',
               'SY2002506': '张念玲', 'SY2002510': '陈则非',
               'SY2002516': '刘汉宏', 'SY2003109': '阚哿',
               'SY2003531': '周锦昊', 'SY2003603': '谢永炫',
               'SY2003704': '刘强', 'SY2005320': '武鹏',
               'SY2006304': '马伯乐', 'SY2041102': '陈序航',
               'SY2041106': '范涛', 'SY2041107': '郝作磊',
               'SY2041117': '宋鸿伟', 'SY2041128': '朱耘谷',
               'SY2041130': '韩敬霖', 'ZY2002114': '洪安琪',
               'ZY2002120': '赖家键', 'ZY2002322': '马超',
               'ZY2002327': '王仔豪', 'ZY2002403': '于沛宏',
               'ZY2002412': '程小宝', 'ZY2002613': '王宇豪',
               'ZY2041131': '周宇航'}
    ID = ['01', 'PT2000005', 'PT2000006', 'PT2000012', 'PT2000054',
           'PT2000437', 'P20000188', 'SY1906903', 'SY2002105', 'SY2002107',
           'SY2002110', 'SY2002112', 'SY2002114', 'SY2002406', 'SY2002506',
           'SY2002510', 'SY2002516', 'SY2003109', 'SY2003531', 'SY2003603',
           'SY2003704', 'SY2005320', 'SY2006304', 'SY2041102', 'SY2041106',
           'SY2041107', 'SY2041117', 'SY2041128', 'SY2041130', 'ZY2002114',
           'ZY2002120', 'ZY2002322', 'ZY2002327', 'ZY2002403', 'ZY2002412',
           'ZY2002613', 'ZY2041131']
    path = 'image_database/'
    processor = picProcessor(ID_name, ID, path)
    faceData = processor.generateData()
    label = processor.generateLabel()
    pca = PCA(1510)
    numOfPattern = max(label) + 1
    numOfSamples = len(faceData)
    K = 10  # number of folds

    faceData = pca.fit_transform(faceData)
    faceDataTensor = torch.tensor(faceData)
    labelTensor = torch.tensor(label)
    randomIndex = torch.randperm(numOfSamples)
    faceDataFoldList = []
    labelFoldList = []
    for i in range(K):
        faceDataFoldList.append(faceDataTensor[randomIndex[int(i * numOfSamples / K): int((i+1) * numOfSamples / K)], :])
        labelFoldList.append(labelTensor[randomIndex[int(i * numOfSamples / K): int((i+1) * numOfSamples / K)]])

    bestModel = None
    minLoss = 1e5
    maxRecog = 0
    for i in range(K):
        if i == 0:
            head = 1
            trainData = faceDataFoldList[1]
            trainLabel = labelFoldList[1]
        else:
            head = 0
            trainData = faceDataFoldList[0]
            trainLabel = labelFoldList[0]
        for j in range(K):
            if j == i:
                continue
            elif j == head:
                continue
            else:
                trainData = torch.cat((trainData, faceDataFoldList[j]), dim=0)
                trainLabel = torch.cat((trainLabel, labelFoldList[j]), dim=0)
        validData = faceDataFoldList[i]
        validLabel = labelFoldList[i]

        model = nnBaseModel(trainData, trainLabel, [1510, 500, 250, 125, numOfPattern])
        model.train_bfgs(200, 0.001)
        validLoss = nn.CrossEntropyLoss()(model(validData), validLabel)
        result = torch.max(model(validData), dim = 1)[1]

        if validLoss < minLoss:
            minLoss = validLoss
            bestModel = model
            torch.save(model, 'bestModel.pkl')
            maxRecog = sum(result.eq(validLabel)) / len(validLabel)

    print(maxRecog)
