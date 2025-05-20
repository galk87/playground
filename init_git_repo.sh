#!/bin/bash
# Initialize a git repository with the production-ready scoreboard recognition system

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
  echo "Initializing git repository..."
  git init
else
  echo "Git repository already initialized."
fi

# Create .gitignore file if it doesn't exist
if [ ! -f ".gitignore" ]; then
  echo "Creating .gitignore file..."
  cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Logs
*.log
logs/

# Cache directories
.cache/
.pytest_cache/

# Results and data
*_results/
*_result.json
batch_results.json
test_images.txt
results.json
single_result.json

# Training data (keep only the necessary examples)
combined_valid_dataset/
unified_dataset/
!continued_iterations_results/best_examples/

# Editor files
.vscode/
.idea/
*.sublime-project
*.sublime-workspace

# OS files
.DS_Store
Thumbs.db
EOL
fi

# Create requirements.txt file if it doesn't exist
if [ ! -f "requirements.txt" ]; then
  echo "Creating requirements.txt file..."
  cat > requirements.txt << 'EOL'
google-generativeai>=0.5.0
Pillow>=10.0.0
tqdm>=4.65.0
EOL
fi

# Make sure the examples directory exists
mkdir -p continued_iterations_results/best_examples/

# Add all necessary files to git
echo "Adding files to git..."
git add scoreboard_recognition.py
git add demo_scoreboard.py
git add batch_test.py
git add README.md
git add requirements.txt
git add .gitignore
git add continued_iterations_results/best_examples/
git add init_git_repo.sh

# Show status
git status

echo ""
echo "=== Next steps ==="
echo "1. Review the added files with 'git status'"
echo "2. Commit the changes with: git commit -m \"Initial commit of scoreboard recognition system\""
echo "3. Add a remote repository: git remote add origin <your-repository-url>"
echo "4. Push to the repository: git push -u origin main"
echo ""
echo "To run these commands, execute:"
echo "git commit -m \"Initial commit of scoreboard recognition system\""
echo "git remote add origin <your-repository-url>"
echo "git push -u origin main" 