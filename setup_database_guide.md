# 🗄️ AquaGenomeAI Database Setup Guide

## Quick Setup

After starting Docker, you need to:
1. Create the database
2. Create collections
3. Create vector index
4. Populate with reference data

## Step 1: Start ArangoDB

```bash
docker compose up -d
```

Wait ~10 seconds for the database to start, then verify:
```bash
docker ps  # Should show aquagenome_db running
```

## Step 2: Create Database & Collections

### Option A: Using ArangoSH (Docker)

```bash
# Access ArangoDB shell
docker exec -it aquagenome_db arangosh

# In ArangoShell, run:
db._createDatabase("AquaGenome");
db._useDatabase("AquaGenome");

// Create collections
db._create("sequence");
db._create("taxon");
db._create("sample");
db._createEdgeCollection("sequence-taxon");

// Create vector index for DNABERT embeddings
db.sequence.ensureIndex({
    name: "dnabert_cosine",
    type: "vector",
    fields: ["embedding"],
    params: { metric: "cosine", dimension: 768, nLists: 100 }
});

// Exit
exit
```

### Option B: Using Python Script

Create `scripts/init_database.py`:

```python
from arango import ArangoClient

# Connect
client = ArangoClient(hosts='http://localhost:8529')
sys_db = client.db('_system', username='root', password='openSesame')

# Create database
if not sys_db.has_database('AquaGenome'):
    sys_db.create_database('AquaGenome')
    print("✅ Created AquaGenome database")

# Use database
db = client.db('AquaGenome', username='root', password='openSesame')

# Create collections
for coll in ['sequence', 'taxon', 'sample']:
    if not db.has_collection(coll):
        db.create_collection(coll)
        print(f"✅ Created {coll} collection")

# Create edge collection
if not db.has_collection('sequence-taxon'):
    db.create_collection('sequence-taxon', edge=True)
    print("✅ Created sequence-taxon edge collection")

# Create vector index
sequence_coll = db.collection('sequence')
sequence_coll.add_index({
    'type': 'vector',
    'name': 'dnabert_cosine',
    'fields': ['embedding'],
    'params': {'metric': 'cosine', 'dimension': 768, 'nLists': 100}
})
print("✅ Created vector index for DNABERT embeddings")

print("\n🎉 Database setup complete!")
```

Run it:
```bash
python scripts/init_database.py
```

## Step 3: Populate with Reference Data

### Download Reference Sequences from NCBI

```python
# scripts/download_references.py
from Bio import Entrez, SeqIO
import os

Entrez.email = "your.email@example.com"

# Download 16S rRNA sequences
query = "16S ribosomal RNA[Title] AND bacteria[Organism]"
handle = Entrez.esearch(db="nucleotide", term=query, retmax=1000)
record = Entrez.read(handle)
id_list = record["IdList"]

# Fetch sequences
handle = Entrez.efetch(db="nucleotide", id=id_list, rettype="fasta", retmode="text")
os.makedirs("data/references", exist_ok=True)

with open("data/references/16S_bacteria_1000.fasta", "w") as f:
    f.write(handle.read())

print(f"✅ Downloaded {len(id_list)} reference sequences")
```

### Process and Insert

```python
# scripts/process_references.py
from Bio import SeqIO
from tools_genomics import GetDNABERTEmbedding
from db import db, sequence_collection
from tqdm import tqdm

# Load FASTA
sequences = list(SeqIO.parse("data/references/16S_bacteria_1000.fasta", "fasta"))

print(f"Processing {len(sequences)} sequences...")

for seq_record in tqdm(sequences):
    sequence = str(seq_record.seq)
    
    # Generate embedding
    embedding = GetDNABERTEmbedding(sequence)
    
    if embedding:
        # Create document
        doc = {
            "_key": f"REF_{seq_record.id}",
            "sequence_id": seq_record.id,
            "sequence": sequence,
            "sequence_type": "16S_rRNA",
            "length": len(sequence),
            "gc_content": (sequence.count('G') + sequence.count('C')) / len(sequence) * 100,
            "embedding": embedding,
            "source": "NCBI",
            "description": seq_record.description
        }
        
        # Insert
        try:
            sequence_collection.insert(doc)
        except:
            pass  # Skip duplicates

print("✅ Reference data loaded!")
```

## Step 4: Verify Setup

```python
from db import db, sequence_collection, taxon_collection

# Check collections
print(f"Sequences: {sequence_collection.count()}")
print(f"Taxa: {taxon_collection.count()}")

# Test vector search
aql = """
FOR seq IN sequence
    FILTER seq.embedding != null
    LIMIT 5
    RETURN {id: seq._key, length: seq.length}
"""
result = list(db.aql.execute(aql))
print(f"\nSample sequences: {result}")
```

## 🎯 You're All Set!

Now you can:
```bash
streamlit run app.py
```

And start analyzing deep-sea sequences! 🌊🧬

---

## Troubleshooting

**Issue:** "Database not found"
- Make sure Docker container is running: `docker ps`
- Recreate database with Step 2

**Issue:** "Vector index not working"
- ArangoDB version must be 3.12+ with `--experimental-vector-index`
- Check docker-compose.yml has the correct command

**Issue:** "Slow embedding generation"
- Use GPU if available (change device in config.yaml)
- Reduce batch size
- Process in batches

**Issue:** "Out of memory"
- Limit number of reference sequences
- Use smaller sequence lengths (< 1000 bp)
- Increase Docker memory limit
