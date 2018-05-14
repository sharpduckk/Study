#-*- coding: utf-8 -*-
# 네이버 Papago NMT API 예제
import os
import sys
import json
import urllib.request
import time
import sqlite3

DB_PATH = "politics.db"
# news 테이블에 값 입력하기
def update_eng_db(eng_news, url):
    con = sqlite3.connect(DB_PATH)
    with con:
        cur = con.cursor()
        try:
            cur.execute("UPDATE news SET eng_contents = ? WHERE url = ?", (eng_news, url))
            con.commit()
        except:
            print("except news:", eng_news)
            print("except url:", url)

def update_contents_len_db(url, length):
    con = sqlite3.connect(DB_PATH)
    with con:
        cur = con.cursor()
        try:
            cur.execute("SELECT url, contents FROM news WHERE url = ? ", (url,))
            sql_result = cur.fetchall()
        except:
            print("select fail: ")
            print("url: ", url)

        # contents with under 400
        string400 = sql_result[0][1][0:401]
        try:
            cur.execute("""UPDATE news SET contents = ? WHERE url = ?""", (string400, url,))
            con.commit()
        except:
            print("except news:", "d")
            print("except url:", url)
    pass

def naver_translate(text):
    client_id = "iXLO2cKvvqiz_5oYsHIZ"
    client_secret = "E3NkELxNFG"
    encText = urllib.parse.quote(text) # text in
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        # print(response_body.decode('utf-8'))
        return response_body.decode('utf-8')
    else:
        return "Error Code:" + rescode


with open('news.json', encoding='UTF8') as data_file:
    data = json.load(data_file)

"""
# set under 400 words in text
for article in data:
    update_contents_len_db(article['url'], 400)

"""

# pprint(data[0]['contents'])
Corpus_set = []
Eng_corpus_set = []
count = 0
# translate to english (naver NMT) (It just takes 10,000 char each app)
# for article in data[220:230]: # because of err, update data divided to many set
for article in data:
    # Corpus_set.append(article['contents'])
    #print(article['contents'])
    eng_text = naver_translate(article['contents'])
    eng_text = json.loads(eng_text)
    eng_text = eng_text['message']['result']['translatedText']
    # print(eng_text, article['url'])
    update_eng_db(eng_text, article['url'])
    # print(count)
    time.sleep(0.1)
    count += 1

"""
for text in Corpus_set:
    eng_text = naver_translate(text)
    Eng_corpus_set.append(eng_text)
    time.sleep(0.1)
"""

# print(Eng_corpus_set)


