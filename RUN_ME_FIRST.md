# 🚀 Run AquaGenomeAI - Step by Step

## ✅ Pre-flight Checklist

Before running, make sure you have:
- [ ] Docker installed and running
- [ ] Python 3.9+ installed
- [ ] Google Gemini API key ([Get one free here](https://makersuite.google.com/app/apikey))

---

## Step 1: Start Docker Database (30 seconds)

```bash
# Start ArangoDB
docker compose up -d

# Wait 10 seconds for it to start
# Then verify it's running:
docker ps
```

You should see `aquagenome_db` in the list.

**Troubleshooting:**
- If Docker isn't running, start Docker Desktop first
- If port 8529 is busy, stop other ArangoDB instances

---

## Step 2: Create .env File (1 minute)

Create a file named `.env` in the project root with:

```
GOOGLE_API_KEY=your_actual_gemini_api_key_here
ARANGO_HOST=http://localhost:8529
ARANGO_USER=root
ARANGO_PASS=openSesame
```

**Get your Gemini API key:**
1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy and paste it into `.env`

---

## Step 3: Install Python Dependencies (2-3 minutes)

```bash
# If you haven't already, create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies (this will take a few minutes)
pip install -r requirements.txt
```

**Note:** First time will download DNABERT-2 model (~450MB). Be patient!

---

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

---

## Step 5: Rename Tools File (5 seconds)

```bash
# Windows:
move tools_genomics.py tools.py

# Linux/Mac:
mv tools_genomics.py tools.py
```

---

## Step 6: Launch AquaGenomeAI! 🚀

```bash
streamlit run app.py
```

Your browser should automatically open to http://localhost:8501

---

## 🧪 Test It's Working

Try these queries in the chat:

### Test 1: Check Tools
```
What genomic analysis tools do you have?
```

Expected: List of 10 tools

### Test 2: Generate Embedding
```
Generate an embedding for the sequence ATCGATCGATCGATCG
```

Expected: Returns a 768-dimensional vector

### Test 3: Database Query
```
How many sequences are in the database?
```

Expected: Should work (will be 0 initially)

### Test 4: NCBI Integration
```
Can you fetch some bacterial sequences from NCBI?
```

Expected: Should offer to download sequences

---

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Streamlit opens in your browser
- ✅ You see "Welcome to AquaGenomeAI! 🌊🧬"
- ✅ The chat interface responds to queries
- ✅ Tools are listed when you ask about them
- ✅ No error messages in the terminal

---

## ⚠️ Common Issues

### "Cannot connect to ArangoDB"
```bash
# Check if Docker is running
docker ps

# If not, restart it
docker compose down
docker compose up -d
```

### "GOOGLE_API_KEY not found"
- Make sure `.env` file exists in the project root
- Check the API key is correct (no quotes needed)
- Restart the Streamlit app after creating `.env`

### "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
# Then reinstall
pip install -r requirements.txt
```

### "DNABERT model download is slow"
- First run downloads ~450MB
- Just wait, it only happens once
- Models are cached in `~/.cache/huggingface/`

### "Port 8501 already in use"
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

---

## 🛑 To Stop Everything

```bash
# Stop Streamlit
# Press Ctrl+C in the terminal

# Stop Docker
docker compose down

# Deactivate virtual environment
deactivate
```

---

## 📊 What to Expect (First Run)

**Database:** Empty (0 sequences, 0 taxa)
- You'll need to add reference data
- Use NCBI download tools
- Or upload your own FASTA files

**Response Time:**
- First embedding: ~5-10 seconds (model loading)
- After that: ~1-2 seconds per sequence
- Queries: <1 second

**Memory Usage:**
- DNABERT model: ~1-2GB RAM
- ArangoDB: ~200MB
- Total: ~2-3GB minimum

---

## 🎯 Next Steps After Launch

1. **Add Reference Data:**
   - Download from NCBI: "Download 100 bacterial 16S sequences"
   - This gives you something to search against

2. **Try Analysis:**
   - Upload your own sequences
   - Run similarity searches
   - Cluster unknowns

3. **Customize:**
   - Edit `config.yaml` for your needs
   - Adjust thresholds
   - Configure clustering parameters

---

Ready to launch? Start with **Step 1** above! 🚀
