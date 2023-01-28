import pandas as pd
df = pd.read_csv("https://docs.google.com/spreadsheets/d/16OlZEn2__3ALyECxYS00vlkECWGORTsfKs3ygWjaue8/export?gid=0&format=csv")[["noun_phrase", "Core concept"]]
def addIndividual(currentClass, superClass): 
    return """
    <owl:Class rdf:about="http://www.semanticweb.org/AI-law#{}">
        <rdfs:subClassOf rdf:resource="http://www.semanticweb.org/AI-law#{}"/>
    </owl:Class>
    """.format(currentClass, superClass)
rdfIndividuals = ""
for i, row in df.iterrows(): 
    if not str(row["Core concept"]) == "nan":
        rdfIndividuals+= addIndividual(str(row["noun_phrase"]).replace(" ", "_"), str(row["Core concept"]).replace(" ", "_"))
with open("onto.rdf", "w") as f:
    f.write(rdfIndividuals)
df.dropna()[["Core concept"]].value_counts()