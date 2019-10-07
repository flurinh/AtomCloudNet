import requests
from tqdm import tqdm
import time



PATH = 'http://www.rcsb.org/pdb/resultsV2/sids.jsp?qrid=8FF875E0'
prefix = 'https://files.rcsb.org/download/'
postfix = '_cs.str'
postfix_pdb = '.pdb'

re = requests.get(PATH, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:63.0) Gecko/20100101 Firefox/63.0'})

content = re.text
list_ids = content.split('\n')
data = []
save_path = 'data/'
print("Checking "+str(len(list(list_ids)))+" proteins...")
n_prots = 0
for i, ID in enumerate(tqdm(list_ids)):
    link = prefix + ID + postfix
    link_pdb = prefix + ID + postfix_pdb
    re = requests.get(link, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:63.0) Gecko/20100101 Firefox/63.0'})
    re_pdb = requests.get(link_pdb, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:63.0) Gecko/20100101 Firefox/63.0'})
    if re.status_code == 200:
        n_prots+=1
        content_ = re.text
        content_pdb = re_pdb.text
        _save_path = save_path+ID+'.txt'
        _save_path_pdb = save_path+ID+'_pdb.txt'
        file = open(_save_path, 'w')
        file.write(content_)
        file_pdb = open(_save_path_pdb, 'w')
        file_pdb.write(content_pdb)
        time.sleep(5)
print("Downloaded a total of "+str(n_prots)+" proteins!")
