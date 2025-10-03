import os
from arango import ArangoClient
from langchain_community.graphs import ArangoGraph
from dotenv import load_dotenv

load_dotenv()
ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASS = os.getenv("ARANGO_PASS", "openSesame")

# ================= AquaGenomeAI Database =================

client = ArangoClient(hosts=ARANGO_HOST)
db = client.db('AquaGenome', username=ARANGO_USER, password=ARANGO_PASS)
print(f"✅ Connected to ArangoDB: {db.name}")

# Initialize LangChain Graph interface
arango_graph = ArangoGraph(db)

# ================= Collections =================

# Core collections
sequence_collection = db.collection('sequence')
taxon_collection = db.collection('taxon')
sample_collection = db.collection('sample')

# Edge collections (relationships)
seq_taxon_link = db.collection('sequence-taxon')

print("✅ ArangoGraph initialized successfully!")
print(f"   - Sequences: {sequence_collection.count()} records")
print(f"   - Taxa: {taxon_collection.count()} records")
print(f"   - Samples: {sample_collection.count()} records")