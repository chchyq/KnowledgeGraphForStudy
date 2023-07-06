import os
import pandas as pd
import re
if __name__ == '__main__':
    file_list = list()

    # csv文件所在目录
    # files = os.listdir("data/wikifier/")
    files=[f for f in os.listdir("data/wikifier/") if f.endswith('.csv')]
    # print(files)
    num=0
    for file in files:
        df = pd.read_csv('data/wikifier/'+file,index_col=0,header=0)
        if('Update' in file):
            pattern=r'\sUpdate\s|.csv'
            result=re.split(pattern,file)
            result[1]='+'+result[1]   
        elif('Paper Topics.csv' in file):
            pattern=r'Paper Topics.csv'
            result=re.split(pattern,file)
            result[1]='-'
        else:
            pattern=r'\sLecture\s|.csv'
            result=re.split(pattern,file)
        # print(result)
        df['Course'] = str(result[0])
        df['LectureNum'] = str(result[1])
        df['Lecture']=re.split(r'.csv',file)[0]
        df['LectureId']=num
        num+=1
        df.reset_index(inplace=True)
        # print(df)
        df['title'].astype(str)
        # print(df.dtypes)
        df = df.drop(df[df['title']=='title'].index)
        file_list.append(df)

    # file_list.sort(key='Course')
    # for i in file_list:
    #     print(i['Course'],i['Lecture'])
    # file_list=pd.DataFrame(file_list)
    # print(file_list)

    all_VA = pd.concat(file_list,axis=0,ignore_index=False)
    all_VA['title']=all_VA['title'].astype("string")
    all_VA['Course']=all_VA['Course'].astype("string")
    all_VA['Lecture']=all_VA['Lecture'].astype("string")
    all_VA['pageRank']=all_VA['pageRank'].astype("float64")
    all_VA.rename(columns={'title':'entity'},inplace=True)
    all_VA.rename(columns={'Unnamed: 0':'entityID'},inplace=True)
    print(all_VA.dtypes)
    all_VA.to_csv("data/all_file.csv")
