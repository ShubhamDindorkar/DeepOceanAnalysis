# 🚀 AquaGenomeAI - Quick Start Guide

Get up and running with AquaGenomeAI in 5 minutes!

## Prerequisites ✅

- [x] Python 3.9+
- [x] Docker
- [x] Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

## Step 1: Start the Database (30 seconds)

```bash
# Start ArangoDB with Docker
docker compose up -d

# Wait for it to start
sleep 10

# Verify it's running
docker ps
```

You should see `aquagenome_db` running.

## Step 2: Set Up Python Environment (2 minutes)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies (this will take a minute)
pip install -r requirements.txt
```

## Step 3: Configure API Keys (1 minute)

Create a `.env` file:

```bash
# Copy the example
cp .env.example .env

# Edit it with your API key
# On Windows:
notepad .env
# On Linux/Mac:
nano .env
```

**Minimum required:**
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

## Step 4: Initialize Database (30 seconds)

```bash
python scripts/init_database.py
```

You should see:
```
✅ Created 'AquaGenome' database
✅ Created 'sequence' collection
✅ Created 'taxon' collection
...
🎉 Database initialization complete!
```

## Step 5: Launch AquaGenomeAI! (10 seconds)

```bash
streamlit run app.py
```

Your browser should open to `http://localhost:8501` 🌊

## First Steps 🧬

Try these example queries:

1. **"What genomic analysis tools do you have?"**
   - See all available tools

2. **"Find sequences similar to ATCGATCGATCG"**
   - Test similarity search (will be empty at first)

3. **"Download some reference sequences from NCBI"**
   - Use: `FetchNCBISequences("deep sea bacteria 16S", max_results=10)`

4. **"Help me analyze a DNA sequence"**
   - Paste a sequence and get it analyzed

## What's Next?

### Add Reference Data

To make the system useful, add reference sequences:

1. **Download from NCBI:**
   ```python
   # In the chat interface:
   "Download 100 bacterial 16S sequences from NCBI"
   ```

2. **Or use the setup guide:**
   See `setup_database_guide.md` for detailed instructions

### Explore Features

- 🔍 **Similarity Search**: Find matching sequences
- 📊 **Clustering**: Discover novel species
- 📚 **Literature**: Search papers with Exa (requires API key)
- 🧬 **Embeddings**: Generate DNABERT-2 representations

### Customize

Edit `config.yaml` to:
- Change similarity thresholds
- Adjust clustering parameters
- Configure BLAST settings
- Set model preferences

## Troubleshooting 🔧

### "Database not found"
```bash
# Restart Docker
docker compose down
docker compose up -d

# Re-run init script
python scripts/init_database.py
```

### "DNABERT model download is slow"
- First time will download ~450MB model
- Be patient, it only happens once
- Models are cached in `~/.cache/huggingface/`

### "Streamlit won't start"
```bash
# Check if port 8501 is busy
# Windows:
netstat -ano | findstr :8501
# Linux/Mac:
lsof -i :8501

# Use different port:
streamlit run app.py --server.port 8502
```

### "Docker container keeps restarting"
```bash
# Check logs
docker logs aquagenome_db

# Common fix: Remove old volumes
docker compose down -v
docker compose up -d
```

## Need Help?

1. Check `CONVERSION_GUIDE.md` for technical details
2. Review `setup_database_guide.md` for database setup
3. Read `README.md` for full documentation
4. Open an issue on GitHub

---

**You're ready to explore the genomic frontier! 🌊🧬**

Happy analyzing!
