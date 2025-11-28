import os
import xmltodict


def parse_xml_files(directory):
    xml_files = []
    # 遍历指定目录下的所有文件
    for root, dirs, files in os.walk(directory):
        files = sorted(files)
        for file in files:
            # 检查文件扩展名是否为.xml
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                xml_files.append(file_path)
    return xml_files


pmids = []
pmids.append('34380729')
pmids.append('34583332')
pmids.append('34835103')
pmids.append('35085865')
pmids.append('35138912')
# 指定要遍历的目录
directory_path = './ebola/'
xml_files = parse_xml_files(directory_path)
for xml in xml_files:
    pmid = xml[8:-7]
    pmids.append(pmid)
    print(int(pmid), end=', ')

print()

from Bio import Entrez, Medline


def cmp(x):
    return int(x)


Entrez.email = "123@example.com"
term = "ebola[Title/Abstract]"

# 1976.01.01-2023.01.01
handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="text")
records = list(Medline.parse(handle))
num = 1
for record in records:
    title = record['TI'] if 'TI' in record else 'nan'  # 标题
    abtxt = record['AB'] if 'AB' in record else 'nan'  # 摘要
    print(str(num) + '.', title)
    num += 1
