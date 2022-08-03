# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 09:19:49 2022

@author: Jikitsha Sheth

"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from itertools import chain

#To read the dataset from CSV file, dataset available from https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
df=pd.read_csv("C:/Users/js/Downloads/Groceries_dataset.csv")


#To create each item's row as list so that further the memberid and datewise the items can be merged
df.itemDescription = df.itemDescription.transform(lambda x: [x])
df1 = df.groupby(['Member_number','Date']).sum()['itemDescription'].reset_index(drop=True)


top5products=df["itemDescription"].value_counts().reset_index(name='Quantity').head(5)

#To encode items so that it can be associated that which item are purchased together
encoder = TransactionEncoder()
itemlist = pd.DataFrame(encoder.fit(df1).transform(df1), columns=encoder.columns_)
#print(itemlist)

#Minimum support is set with respect to items in the data frame i.e. overall transaction count
minSupport=10/len(df1)

#I am interested to find only till 2-frequent itemset and based on that rules are extracted with lift 1.5
frequent_itemsets = apriori(itemlist, min_support= minSupport, use_colnames=True, max_len = 2)
rules = association_rules(frequent_itemsets, metric="lift",  min_threshold = 1.5)



#Function that returns list of related items based on rules generated
def findAssociateditems(x):
    new_rules = rules[(rules['antecedents'].astype(str).str.contains(x))]
    new_rules = new_rules.sort_values(by=['lift'],ascending = [False]).reset_index(drop = True)
    return new_rules['consequents']


def frontend():
    member_id=int(input("Enter member id : "))
    print(member_id)
    date_of_transaction=input("Enter date (in DD-MM-YYYY) ") 
    print(date_of_transaction)
    #member_id=1808
    #date_of_transaction="21-07-2015"
    
    
    itemsfound=df[(df['Member_number'] == member_id) & (df['Date']==date_of_transaction)]
    
    #If record is found then find associated items, else show top 5 purchased items
    if(len(itemsfound)>0):
        print("Member", member_id, "on date", date_of_transaction, "has purchased", list(chain.from_iterable(list(itemsfound['itemDescription']))))
        resultlist=[]
        for item in itemsfound['itemDescription']:
            resultlist.append(findAssociateditems(str(item[0])))
        
        #To convert the list of series of set into flattened list
        prodlist=[]
        prodset=set()
        for item in resultlist:
            for i in item:
                prodlist.append(list(i))
        prodlist=list(chain.from_iterable(prodlist))
        prodset=set(prodlist)#To remove duplicate items if any
        prodlist=list(prodset)          
        print("Products recommended are",prodlist)

    else:
        print("No such member or date found in dataset")
        print("However, we recommend you to purchase following top products:")
        print(list(chain.from_iterable(list(top5products['index']))))
        
         
frontend()
