import networkx as nx
import json

with open("data/processed/drug_properties.json") as f:
    prop_db = json.load(f)

G = nx.Graph()

def build_graph():

    for drug, props in prop_db.items():
        for p in props:
            G.add_edge(drug, p)

    return G

graph = build_graph()

def get_reasoning_path(d1, d2):

    if d1 in graph and d2 in graph:
        try:
            return nx.shortest_path(graph, d1, d2)
        except:
            return []

    return []