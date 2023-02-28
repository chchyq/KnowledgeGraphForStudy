import json
import csv
import itertools
import urllib
from string import punctuation
import nltk
import os
# 防止字符串乱码
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

# ENTITY_TYPES = ["human", "person", "company", "enterprise", "business", "geographic region",
#                 "human settlement", "geographic entity", "territorial entity type", "organization"]

def wikifier(text, lang="en", threshold=0.8):
    """Function that fetches entity linking results from wikifier.com API"""
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", "tgbdmkpmkluegqfbawcwjywieevmza"),
        ("pageRankSqThreshold", "%g" %
         threshold), ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "100"), ("nWordsToIgnoreFromList", "100"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
        ("includeCosines", "false"), ("maxMentionEntropy", "3")
    ])
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout=60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))
    # Output the annotations.
    results = list()
    # print(response)
    for annotation in response["annotations"]:
        results.append({'title': annotation['title'],  'pageRank': annotation['pageRank']})#'wikiId': annotation['wikiDataItemId'],'characters': [(el['chFrom'], el['chTo']) for el in annotation['support']]
    return results


if __name__ == '__main__':
    # res=wikifier("Elon Musk is a business magnate, industrial designer, and engineer. He is the founder, CEO, CTO, and chief designer of SpaceX. He is also early investor, CEO, and product architect of Tesla, Inc. He is also the founder of The Boring Company and the co-founder of Neuralink. A centibillionaire, Musk became the richest person in the world in January 2021, with an estimated net worth of $185 billion at the time, surpassing Jeff Bezos. Musk was born to a Canadian mother and South African father and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada aged 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received dual bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University, but decided instead to pursue a business career. He went on co-founding a web software company Zip2 with his brother Kimbal Musk.")

    # print(res)
    txt_path='/Users/chenyingqing/Documents/Internship/course1/'
    wikifier_path='/Users/chenyingqing/Documents/Internship/wikifier1/'
    txt_lst = [f for f in os.listdir(txt_path) if f.endswith('.txt')]
    # txt_lst = [os.path.join(txt_path, filename) for filename in txt_lst]
    for i in txt_lst:
        fi=os.path.join(txt_path, i)
        with open(fi,"r") as f:
            data=f.read(10000)
            WikifierList=wikifier(data)
            data=f.read(10000)
            while(data):
                res=wikifier(data)
                for item in res:
                    exist=1
                    for j in WikifierList:
                        if(j['title']==item['title']):
                            exist=0
                            if(j['pageRank']<item['pageRank']):
                                j=item
                            break
                    if(exist):
                        WikifierList.append(item)
                data=f.read(10000)
                # print(res)
            WikifierList=sorted(WikifierList, key = lambda i: i['pageRank'],reverse=True)
            print(WikifierList)
            title=[]
            pageRank=[]
            for w in WikifierList:
                title.append(w['title'])
                pageRank.append(w['pageRank'])
            l=['title','pageRank']
            choose = 1
            for m in range(len(title)):
                all=[title[m],pageRank[m]]
                with open(wikifier_path+i+'.csv', 'a+',newline='', encoding='utf8') as csvfile:
                    writer = csv.writer(csvfile)
                    if choose == 1:
                        writer.writerow(l)
                        choose = 10
                    writer.writerow(all)
