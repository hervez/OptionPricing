from bs4 import BeautifulSoup
import requests
import datetime
import time 

#class WebScrapper():

def get_datestamp():  
    today = int(time.time()) 
    # print(today)  
    date = datetime.datetime.fromtimestamp(today)  
    yy = date.year  
    mm = date.month  
    dd = date.day
    dd += 1
    options_day = datetime.date(yy, mm, dd) 
    datestamp = int(time.mktime(options_day.timetuple())) 
    # print(datestamp) # print(datetime.datetime.fromtimestamp(options_stamp))

    return datestamp 

datestamp = get_datestamp() 
url = "https://finance.yahoo.com/quote/SPY/options?date=" + str(datestamp)
response = requests.get(url)
html_content = response.content
print(html_content)

soup = BeautifulSoup(html_content, "html.parser")
options_tables = soup.find_all("table")
options_tables = [] 
tables = soup.find_all("table")
for i in range(0, len(soup.find_all("table"))):
    options_tables.append(tables[i])
print(options_tables)

expiration = datetime.datetime.fromtimestamp(int(datestamp)).strftime("%Y-%m-%d")
calls = options_tables[0].find_all("tr")[1:]
itm_calls = []
otm_calls = []
for call_option in calls:    
    if "in-the-money" in str(call_option):
        itm_calls.append(call_option)  
    else:    
        otm_calls.append(call_option)

itm_call = itm_calls[-1]
otm_call = otm_calls[0]
print(str(itm_call) + " \n\n " + str(otm_call))
itm_call_data = [] 
for td in BeautifulSoup(str(itm_call), "html.parser").find_all("td"):
    itm_call_data.append(td.text)

itm_call_info = {"contract": itm_call_data[0], "strike": itm_call_data[2], "last": itm_call_data[3],  "bid": itm_call_data[4], "ask": itm_call_data[5], "volume": itm_call_data[8], "iv": itm_call_data[10]}
print(itm_call_info)

callotm_call_data = []
for td in BeautifulSoup(str(otm_call), "html.parser").find_all("td"):  
    otm_call_data.append(td.text)
print(otm_call_data)
otm_call_info = {"contract": otm_call_data[0], "strike": otm_call_data[2], "last": otm_call_data[3],  "bid": otm_call_data[4], "ask": otm_call_data[5], "volume": otm_call_data[8], "iv": otm_call_data[10]}
print(otm_call_info)