#!/usr/bin/env python3
"""
Initialize AquaGenomeAI Database
================================

Creates database, collections, and vector index in ArangoDB.
"""

import sys
from arango import ArangoClient
from dotenv import load_dotenv
import os

load_dotenv()

ARANGO_HOST = os.getenv("ARANGO_HOST", "http://localhost:8529")
ARANGO_USER = os.getenv("ARANGO_USER", "root")
ARANGO_PASS = os.getenv("ARANGO_PASS", "openSesame")

def init_database():
    """Initialize the AquaGenome database with all collections."""
    
    try:
        # Connect to system database
        client = ArangoClient(hosts=ARANGO_HOST)
        sys_db = client.db('_system', username=ARANGO_USER, password=ARANGO_PASS)
        
        print("🔌 Connected to ArangoDB")
        
        # Create database if it doesn't exist
        if not sys_db.has_database('AquaGenome'):
            sys_db.create_database('AquaGenome')
            print("✅ Created 'AquaGenome' database")
        else:
            print("ℹ️  'AquaGenome' database already exists")
        
        # Connect to AquaGenome database
        db = client.db('AquaGenome', username=ARANGO_USER, password=ARANGO_PASS)
        
        # Create document collections
        collections = ['sequence', 'taxon', 'sample']
        for coll_name in collections:
            if not db.has_collection(coll_name):
                db.create_collection(coll_name)
                print(f"✅ Created '{coll_name}' collection")
            else:
                print(f"ℹ️  '{coll_name}' collection already exists")
        
        # Create edge collection
        if not db.has_collection('sequence-taxon'):
            db.create_collection('sequence-taxon', edge=True)
            print("✅ Created 'sequence-taxon' edge collection")
        else:
            print("ℹ️  'sequence-taxon' edge collection already exists")
        
        # Create vector index for DNABERT embeddings
        sequence_coll = db.collection('sequence')
        
        # Check if index exists
        indexes = sequence_coll.indexes()
        has_vector_index = any(idx.get('type') == 'vector' for idx in indexes)
        
        if not has_vector_index:
            try:
                sequence_coll.add_index({
                    'type': 'vector',
                    'name': 'dnabert_cosine',
                    'fields': ['embedding'],
                    'params': {
                        'metric': 'cosine',
                        'dimension': 768,
                        'nLists': 100
                    }
                })
                print("✅ Created vector index for DNABERT embeddings (768-dim)")
            except Exception as e:
                print(f"⚠️  Vector index creation failed: {e}")
                print("   Make sure ArangoDB is started with --experimental-vector-index")
        else:
            print("ℹ️  Vector index already exists")
        
        # Create hash indexes for common queries
        if not any(idx.get('fields') == ['sequence_id'] for idx in indexes):
            sequence_coll.add_hash_index(fields=['sequence_id'], unique=False)
            print("✅ Created hash index on sequence_id")
        
        taxon_coll = db.collection('taxon')
        taxon_indexes = taxon_coll.indexes()
        if not any(idx.get('fields') == ['scientific_name'] for idx in taxon_indexes):
            taxon_coll.add_hash_index(fields=['scientific_name'], unique=False)
            print("✅ Created hash index on scientific_name")
        
        print("\n🎉 Database initialization complete!")
        print(f"\n📊 Current stats:")
        print(f"   - Sequences: {sequence_coll.count()}")
        print(f"   - Taxa: {taxon_coll.count()}")
        print(f"   - Samples: {db.collection('sample').count()}")
        print(f"   - Links: {db.collection('sequence-taxon').count()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
