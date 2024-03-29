# 读取pubtator文件，获取pmid

ebola_pmid = list()
with open('./output-testing(1976-2024).txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for line in lines:
        if len(line.split('|')) >= 3 and line.split('|')[1] == 't':
            ebola_pmid.append(line.split('|')[0])

with open('./pmid.txt', 'w', encoding='utf-8') as file:
    for pmid in ebola_pmid:
        file.write(pmid + '\n')

# cdr_pmid = set()
# with open('./cdr/CDR_DevelopmentSet.PubTator.txt', 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#     for line in lines:
#         if len(line.split('|')) >= 3 and line.split('|')[1] == 't':
#             cdr_pmid.add(line.split('|')[0])
# with open('./cdr/CDR_TestSet.PubTator.txt', 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#     for line in lines:
#         if len(line.split('|')) >= 3 and line.split('|')[1] == 't':
#             cdr_pmid.add(line.split('|')[0])
# with open('./cdr/CDR_TrainingSet.PubTator.txt', 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#     for line in lines:
#         if len(line.split('|')) >= 3 and line.split('|')[1] == 't':
#             cdr_pmid.add(line.split('|')[0])
#
# gda_pmid = set()
# with open('./gda/testing.pubtator', 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#     for line in lines:
#         if len(line.split('|')) >= 3 and line.split('|')[1] == 't':
#             gda_pmid.add(line.split('|')[0])
# with open('./gda/training.pubtator', 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#     for line in lines:
#         if len(line.split('|')) >= 3 and line.split('|')[1] == 't':
#             gda_pmid.add(line.split('|')[0])
#
# set1 = ebola_pmid.union(cdr_pmid)
# set2 = ebola_pmid.union(gda_pmid)
# print(len(ebola_pmid) + len(cdr_pmid) - len(set1))
# print(len(ebola_pmid) + len(gda_pmid) - len(set2))
# print()
