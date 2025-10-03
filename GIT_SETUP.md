# Git Setup Guide for AquaGenomeAI

## Install Git

### Windows
1. Download Git for Windows: https://git-scm.com/download/win
2. Run the installer
3. **Important**: During installation, select "Git from the command line and also from 3rd-party software"
4. Restart your terminal/PowerShell after installation

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install git
```

### Mac
```bash
brew install git
```

## Initial Git Setup

After installing Git, configure it:

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email
git config --global user.email "your.email@example.com"

# Verify
git config --list
```

## Initialize Repository (First Time)

If this is a new repository:

```bash
# Initialize git
git init

# Add remote (if you have a GitHub repo)
git remote add origin https://github.com/your-username/AquaGenomeAI.git
```

## Make Your First Commit

### Option 1: Using the Helper Script

**Windows:**
```bash
# Run the batch script
git_commit.bat
```

**Linux/Mac:**
```bash
# Make script executable
chmod +x git_commit.sh

# Run it
./git_commit.sh
```

### Option 2: Manual Commit

```bash
# Stage all changes
git add -A

# Commit with the prepared message
git commit -F COMMIT_MESSAGE.txt

# Or write your own message
git commit -m "🌊 Initial release: AquaGenomeAI deep-sea genomic platform"
```

## Push to GitHub

```bash
# First time (set upstream)
git push -u origin main

# After that
git push
```

## Common Git Commands

```bash
# Check status
git status

# View commit history
git log

# View last commit
git log -1

# Create a new branch
git checkout -b feature/new-tool

# Switch branches
git checkout main

# Pull latest changes
git pull origin main

# View what changed
git diff
```

## .gitignore Already Configured

The `.gitignore` file is already set up to exclude:
- Large data files (FASTA, FASTQ)
- Embeddings and FAISS indices
- BLAST databases
- API keys (.env)
- Python cache
- Logs

## Troubleshooting

### "Git is not recognized"
- Make sure Git is installed
- Restart your terminal
- Check PATH environment variable includes Git

### "Permission denied (publickey)"
- Set up SSH keys for GitHub
- Or use HTTPS with personal access token

### "Nothing to commit"
- You might have already committed
- Check `git status`

### "Failed to push"
- Make sure you have write access to the repo
- Try `git pull` first to sync

## Best Practices

1. **Commit often** with clear messages
2. **Don't commit** large data files (already in .gitignore)
3. **Don't commit** API keys (they're in .gitignore)
4. **Use branches** for new features
5. **Pull before push** to avoid conflicts
6. **Review changes** before committing: `git status`

## Example Workflow

```bash
# 1. Make changes to your code
# (edit files...)

# 2. Check what changed
git status
git diff

# 3. Stage changes
git add -A

# 4. Commit
git commit -m "Add new clustering visualization tool"

# 5. Push
git push origin main
```

## GitHub Setup (Optional)

If you want to host on GitHub:

1. Create a new repository on GitHub
2. **Don't** initialize with README (we already have one)
3. Copy the repository URL
4. Add as remote:
   ```bash
   git remote add origin https://github.com/your-username/AquaGenomeAI.git
   ```
5. Push:
   ```bash
   git push -u origin main
   ```

---

## Ready to Commit?

Run the helper script:
- **Windows**: `git_commit.bat`
- **Linux/Mac**: `./git_commit.sh`

Or commit manually:
```bash
git add -A
git commit -F COMMIT_MESSAGE.txt
```

The commit message is already prepared in `COMMIT_MESSAGE.txt` without any mention of the previous project! 🌊🧬
