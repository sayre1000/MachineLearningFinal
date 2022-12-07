import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/MyDrive/CS461 Final Project"

import requests

base_url = 'https://adventuretime.fandom.com/wiki/'
ep_url = base_url + 'Category_talk%3ATranscripts'

resp = requests.get(ep_url)

from bs4 import BeautifulSoup

content = BeautifulSoup(resp.text,'html.parser')

webpage = content.find(id='mw-content-text')

web_data = webpage.find_all('td')

transcripts = []

for row in web_data:
  row = str(row)
  row = row.split('title=')
  if(len(row) > 1):
    row[1] = row[1].split(">")
    transcripts.append(row[1][0])
transcripts = transcripts[1:]
print(transcripts)
print(len(transcripts))

tran_urls = []
for tran in transcripts:
  tran = tran.replace('"', "");
  tran = tran.replace(" ", "_")
  tran_url = base_url + tran

  tran_urls.append(tran_url)

import re
CLEAN = re.compile('<.*?>')


x_data = []
y_data = []

for url in tran_urls:
  tran_resp = requests.get(url)

  tran_content = BeautifulSoup(tran_resp.text,'html.parser')
  tran_content = tran_content.find_all('dl')

  num_lines = len(tran_content)
  curr_line = 0
  
  while(curr_line < num_lines):

    line = str(tran_content[curr_line])
    if "<b>Finn</b>" in line:
      line = re.sub(CLEAN, '', line)
      prev_line = str(tran_content[curr_line - 1])
      if ":" in prev_line:
        prev_line = re.sub(CLEAN,'',prev_line)
      else:
        prev_line = re.sub(CLEAN,'',str(tran_content[curr_line - 2]))
   
      prev_line = re.sub(r'\[(.*?)\]', '', prev_line)
      if "Finn:" in prev_line:
        prev_line = "Other: Other"
      line =  re.sub(r'\[(.*?)\]', '', line)
     
      if " &amp;" in prev_line:
        prev_line.replace(' &amp;', ':')
      
      line = line.split(':')
      prev_line = prev_line.split(':')

      
      if(len(prev_line) > 1 and len(line) > 1):
        new_line = prev_line[1] + ':' + line[1]
        print(new_line)
        x_data.append(new_line)
  
    curr_line += 1

x_data_split = []
print(len(x_data))
for x in x_data:
  x = x.split(": ")
  x_data_split.append(x)

print(len(x_data_split))

import csv
header = ['input','output']
with open('/content/drive/MyDrive/CS461 Final Project/data.csv', 'w', newline = '\n') as f:
  writer = csv.writer(f, delimiter = '|')
  writer.writerow(header)

  for x in x_data_split:
    print(x)
    writer.writerow(x)
