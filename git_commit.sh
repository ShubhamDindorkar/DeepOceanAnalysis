#!/bin/bash
# AquaGenomeAI - Git Commit Helper

echo "========================================"
echo " AquaGenomeAI - Git Commit Helper"
echo "========================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "[ERROR] Git is not installed"
    echo ""
    echo "Please install Git:"
    echo "  Ubuntu/Debian: sudo apt-get install git"
    echo "  Mac: brew install git"
    exit 1
fi

echo "Staging all changes..."
git add -A

echo ""
echo "Files to be committed:"
git status --short

echo ""
echo "========================================"
read -p "Proceed with commit? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo "Commit cancelled."
    exit 0
fi

echo ""
echo "Creating commit..."
git commit -F COMMIT_MESSAGE.txt

echo ""
echo "========================================"
echo "Commit complete!"
echo ""
echo "Next steps:"
echo "  1. Review with: git log -1"
echo "  2. Push with: git push origin main"
echo "========================================"
echo ""
