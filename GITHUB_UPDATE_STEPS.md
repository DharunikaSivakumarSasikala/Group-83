# Updating the Group-83 GitHub Repository

Repository:
https://github.com/DharunikaSivakumarSasikala/Group-83

## Recommended method on macOS

### 1. Clone the existing repository

```bash
cd ~/Downloads
git clone https://github.com/DharunikaSivakumarSasikala/Group-83.git
cd Group-83
```

### 2. Create a safety branch

```bash
git checkout -b skillscope-v4-update
```

### 3. Replace the old repository working files

Keep the hidden `.git` folder. Remove the existing working files:

```bash
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
```

### 4. Copy the clean v4 files into the repository

If the clean folder is also in Downloads:

```bash
cp -R ../Group-83_SkillScope_v4_clean/. .
```

### 5. Check what will be committed

```bash
git status
```

Confirm that `.venv`, `__pycache__`, `__MACOSX`, and `results/embedding_cache` are NOT listed.

### 6. Commit the update

```bash
git add .
git commit -m "Add SkillScope AI v4 prototype and update README"
```

### 7. Push the branch

```bash
git push -u origin skillscope-v4-update
```

You can then merge `skillscope-v4-update` into `main` on GitHub after checking the files.

## If you want to push directly to main instead

After cloning:

```bash
git checkout main
git pull origin main
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -R ../Group-83_SkillScope_v4_clean/. .
git add .
git status
git commit -m "Add SkillScope AI v4 prototype and update README"
git push origin main
```

The branch method is safer because it lets the team review the replacement before merging.
