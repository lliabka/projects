# -*- coding: utf-8 -*-
"""
Created on Wed May 11 18:55:17 2022

@author: Professional
"""
import requests
from bs4 import BeautifulSoup as BS
import MySQLdb

count_words = 10000
arr_defs = []
arr_names = []
for i in range(count_words):
    r = requests.get(f"https://onlinedic.net/ozhegov/page/word{i+1}.php")  
    r.encoding = ('1251')  
    soup = BS(r.text, "lxml")
    title = soup.find('div', class_="word").find('h1').text
    title = title.replace("\r\n\t\t\t\t\t\t","")
    value = soup.find('div', class_="text").text
    value = value.replace("\r\n\t\t\t\t\t\t","")
    arr_defs.insert(i,value)
    arr_names.insert(i,title)
 

db = MySQLdb.connect("localhost","root","","surdo")
insertrec = db.cursor()
for i in range(count_words):
     sqlquery = "insert into words_ozhegov(name,description) values ('"+arr_names[i]+"','"+arr_defs[i]+"')"
     insertrec.execute(sqlquery)
     db.commit()
    
sqlquery = "Select name From words_ozhegov INNER JOIN srd_surd_words ON words_ozhegov.name = srd_surd_words.word"
sqlquery_1 = "Select description From words_ozhegov INNER JOIN srd_surd_words ON words_ozhegov.name = srd_surd_words.word"
insertrec.execute(sqlquery)
query_result = insertrec.fetchall()
insertrec.execute(sqlquery_1)
query_result_1 = insertrec.fetchall()

for i in range(len(query_result)):
    sqlquery = "insert into words_for_markup(name,description) values ('"+str(query_result[i][0])+"','"+str(query_result_1[i][0])+"')"
    insertrec.execute(sqlquery)
    db.commit()


db.close()