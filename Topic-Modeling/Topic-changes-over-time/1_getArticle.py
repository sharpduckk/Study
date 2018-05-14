#-*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import urllib.request
import re, sqlite3
example_url = "http://media.daum.net/breakingnews/politics?regDate=20180401"
# base url  # 04/01 ~ 04/30
politics_url = "http://media.daum.net/breakingnews/politics?regDate="

URL = "http://media.daum.net/ranking/popular/"
DB_PATH = "politics.db"
dict_news = {} # {url: [title, name, write]}


def get_soup(url):
    response = urllib.request.urlopen(url)
    soup = BeautifulSoup(response, "lxml")
    return soup

# 기사에서 기자 성명, 기사 입력 시간 추출
def get_info(href):
    soup = get_soup(href)
    # article info
    info_view = soup.find("span", {"class":"info_view"})
    txt_info = info_view.find_all("span", {"class":"txt_info"})

    name = ""
    write = ""
    pattern = r"\d{4}(\.\d{2}){2}\.? \d{2}:\d{2}"  # 입력 시간 regular expression pattern


    for i in txt_info:
        txt = i.text
        m = re.search(pattern, txt)  # article insert time pattern

        # insert time / reporter
        if not m:  # 기자 성명 추출 (if not insert time pattern)
            name = txt
        elif txt.find("입력") >= 0: # 기사 입력 시간 추출
            write = m.group(0)

            if write[10] == '.': # 기사 입력 시간 형식 맞추기
                write = write[:10] + write[11:]

    # article contents
    contents= ""
    article_view = soup.find("div", {"class": "article_view"})
    section_contents = article_view.find("section")
    txt_contents = section_contents.find_all("p")
    email_pattern =  r"(\w+[\w\.]*)@(\w+[\w\.]*)\.([A-Za-z]+)"
    for j in txt_contents:
        con_txt = j.text
        if not re.search(email_pattern, con_txt):  # article outro pattern
            contents += con_txt

    return name, write, contents

# news list parsing
def set_news(url):
    soup = get_soup(url)

    uls = soup.find("ul", {"class":"list_news2"})
    atags = uls.find_all("a", {"class":"link_txt"})

    for atag in atags:
        title = atag.text
        href = atag['href']
        name, write, contents = get_info(href)

        dict_news[href] = [title, name, write, contents]

# DataBase 생성
def create_db():
    con = sqlite3.connect(DB_PATH)
    with con:
        cur = con.cursor()

        """
        # news 테이블이 존재하면 삭제
        sql_drop = "DROP TABLE IF EXISTS news"
        cur.execute(sql_drop)
        con.commit()
        """
        
        # news 테이블 생성
        sql_create = "CREATE TABLE IF NOT EXISTS news(url text PRIMARY KEY, title text, name text, write DATETIME" \
                     ", contents text, eng_contents text)"
        cur.execute(sql_create)
        con.commit()

# news 테이블에 값 입력하기
def insert_db(news):
    con = sqlite3.connect(DB_PATH)
    with con:
        cur = con.cursor()
        title = dict_news[news][0]
        name = dict_news[news][1]
        write = dict_news[news][2]
        contents = dict_news[news][3]
        try:
            cur.execute("INSERT INTO news VALUES(?,?,?,?,?)", (news, title, name, write, contents))
            con.commit()
        except:
            print("except news:", news)

# main
def main():
    create_db()
    url_key = []
    for day in range(1, 31):  # 1 ~ 30
        day_str = str(day).zfill(2)  # fill front 0
        set_news(politics_url + "201804" + day_str)
        url_key += list(dict_news.keys())  # news url list (To check duplicate url)

        for news in dict_news:
            print(news)
            print(dict_news[news])

            url_key.remove(news)
            if news in url_key:  # after remove, another left in list, if then
                print("[Notice] Duplicated article")
                continue

            if not dict_news[news][3] == "":  # article text not null
                insert_db(news)

if __name__ == "__main__":
    main()
