import pandas as pd
import json
import csv
import itertools
import urllib
from string import punctuation
import nltk
import os
import re
import math
def split_text(text, length):
    text_list = []
    group_num = len(text) / int(length)
    print(group_num)  # 5.333333333333333
    group_num = math.ceil(group_num)  # 向上取整
    for i in range(group_num):
        tmp = text[i * int(length):i * int(length) + int(length)]
        # print(tmp)
        text_list.append(tmp)
    return text_list
if __name__ == '__main__':
    episodes_df = pd.read_csv("data/episodes.csv", sep='|')
    print(f"{episodes_df.shape[0]} sessions describe by {episodes_df.shape[1]} fields")
    print(episodes_df.count())
    print(episodes_df['Transcriptions'][0])
    Lecure=[]
    Transcriptions=[]
    for i in range(57,len(episodes_df['Transcriptions'])):
        # print(episodes_df['Transcriptions'][i])
        # if('Exam' in episodes_df['Number In Series'][i]):
        #     continue
        name=episodes_df['Course Number'][i]+" "+episodes_df['Number In Series'][i]
        Lecure.append(name)
        Transcriptions.append(episodes_df['Transcriptions'][i])
    Lecture = [value for value in Lecure if 'Exam' not in value]
    index= [i for i, x in enumerate(Lecure) if 'Exam' in x ]
    Transcription=[x for i, x in enumerate(Transcriptions) if i not in index]

    l=['Lecture','Transcription']
    choose = 1
    for m in range(len(Lecture)):
        all=[Lecture[m],Transcription[m]]
        with open("data/Transcriptions1.csv", 'a+',newline='', encoding='utf8') as csvfile:
            writer = csv.writer(csvfile)
            if choose == 1:
                writer.writerow(l)
                choose = 10
            writer.writerow(all)
#     wiki=[]
#     for i in range(57,len(episodes_df['Transcriptions'])):
#         if(isinstance(episodes_df['Transcriptions'][i],str)==False):
#             continue
#         str1=split_text(episodes_df['Transcriptions'][i],10000)
#         # str1 = re.findall(r'.{10000}', episodes_df['Transcriptions'][i])
#         WikifierList=[]
#         for j in str1:
#             print(j)
#             res=wikifier(j)
#             for item in res:
#                     exist=1
#                     for j in WikifierList:
#                         if(j['title']==item['title']):
#                             exist=0
#                             if(j['pageRank']<item['pageRank']):
#                                 j=item
#                             break
#                     if(exist):
#                         WikifierList.append(item)
#         WikifierList=sorted(WikifierList, key = lambda j: j['pageRank'],reverse=True)
#         print(WikifierList)
#         title=[]
#         pageRank=[]
#         for w in WikifierList:
#             title.append(w['title'])
#             pageRank.append(w['pageRank'])
#         l=['title','pageRank']
#         choose = 1
#         for m in range(len(title)):
#             all=[title[m],pageRank[m]]
#             with open("data/Lecture/"+episodes_df['Course Number'][i]+" "+episodes_df['Number In Series'][i]+'.csv', 'a+',newline='', encoding='utf8') as csvfile:
#                 writer = csv.writer(csvfile)
#                 if choose == 1:
#                     writer.writerow(l)
#                     choose = 10
#                 writer.writerow(all)

# # chapters_df = pd.read_csv("data/chapters.csv", sep='|')
# # print(f"{chapters_df.shape[0]} chapters describe by {chapters_df.shape[1]} fields")
# # print(chapters_df.count())
# # print(chapters_df.head(5))
# # # print(chapters_df)