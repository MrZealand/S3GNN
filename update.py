from py2neo import Graph
from py2neo import NodeMatcher,Subgraph
import pandas as pd

graph = Graph(localhost, username="neo4j", password="neo4j")
print(graph)

tx=graph.begin()

APPLY={'APPLYNO':["S201705150426111","S201705160428843"]}
APPLY=pd.DataFrame(APPLY)


NAME={'NAME':['xxx','xxx']}
NAME=pd.DataFrame(NAME)
print(APPLY)
def findNode(graph):
    matcher = NodeMatcher(graph)
    m = matcher.match("APPLYNO",APPLYNO=set(APPLY['APPLYNO']))
    return list(m)

nodes=findNode(graph)
for node,i in zip(nodes,range(len(NAME))):
    node['NAME']=NAME.iloc[i]['NAME']
sub=Subgraph(nodes=nodes)
tx.push(sub)
tx.commit()
