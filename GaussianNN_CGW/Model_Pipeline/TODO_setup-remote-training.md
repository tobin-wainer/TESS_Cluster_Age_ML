# Setup Remote Training - Current Status

## 📋 Remote Training System Summary

**Fully automated hyperparameter training on Windows desktop (8-core CPU + GTX 1080 Ti GPU) via Mac notebook**

### How It Works:
1. **Mac Jupyter notebook** → defines parameter grid (`params` dict)
2. **rsync** → syncs code to Windows desktop (via WSL)
3. **tmux session** → starts background training (survives SSH disconnect)
4. **Training runs** → saves results + logs to `TempFiles/`
5. **rsync back** → retrieves results to Mac `RemoteTempFiles/`
6. **Load & analyze** → process results in notebook

### Key Features:
- ✅ **One-click execution** from notebook (`sync_to_remote()` + `start_remote_training(params)`)
- ✅ **Background training** in tmux (non-blocking, persistent)
- ✅ **GPU support** via `use_gpu` flag (20-30% speedup for small models, up to 8x for larger batches)
- ✅ **Real-time monitoring** (`check_remote_status()`, `tail_remote_log()`, `monitor_gpu()`)
- ✅ **Automatic result sync** (`sync_results_back()`)
- ✅ **Kill function** (`kill_remote_training()`) for hung jobs

### Available Functions:
```python
# Setup & Execution
sync_to_remote()                    # Sync repo to remote
start_remote_training(params)       # Start training in tmux
kill_remote_training()              # Kill training session

# Monitoring
check_remote_status()               # Check if training running
tail_remote_log(lines=50)           # View training logs
monitor_gpu('remote', 2, 20)        # Monitor GPU utilization

# GPU Tools
check_gpu_remote()                  # Check GPU status/availability
diagnose_gpu_remote()               # Troubleshoot GPU issues

# Results
sync_results_back()                 # Download results + logs + figures
load_remote_results()               # Load .pkl results for analysis
```

### Performance:
- **CPU (8 cores):** ~16.6 min for 7 folds × 50 epochs
- **GPU (GTX 1080 Ti):** ~13 min for 7 folds × 50 epochs (batch_size=20)
  - Note: Small model + small batch → GPU underutilized
  - For better GPU speedup: use `batch_size=[64-256]` (4-8x faster, may affect convergence)

### Connection:
- **Mac** → SSH → **Windows** → WSL → conda env `ml` → Python
- All commands wrapped in `wsl` prefix
- Password-free via SSH keys (`C:\ProgramData\ssh\administrators_authorized_keys`)

---

## Overview
Setting up automated hyperparameter training on remote desktop (desktop-win) using rsync + tmux.

## 🎉 REMOTE SETUP COMPLETED! (Dec 12, 2024)

### Issue #1: SSH Authentication - RESOLVED ✅

**Problem**: Windows SSH for administrator accounts uses a different `authorized_keys` file location.

**Solution**:
- Standard users: `C:\Users\USERNAME\.ssh\authorized_keys`
- **Administrator users**: `C:\ProgramData\ssh\administrators_authorized_keys` ← **This is what fixed it!**

After placing the public key in the admin-specific location, password-free SSH now works perfectly.

### Issue #2: SSH Lands in Windows CMD, Not WSL - RESOLVED ✅

**Problem**: SSH to desktop-win lands in native Windows CMD, not WSL.
- Unix commands (`tmux`, `conda`, `pwd`, etc.) don't work
- Paths like `~/workspace` don't exist in Windows
- All remote execution code assumed Unix environment

**Solution**: Wrap all commands in `wsl` prefix and use existing `~/projects` directory
- Changed `REMOTE_BASE` from `~/workspace` to `~/projects` (already exists in WSL)
- All SSH commands now wrapped: `ssh user@host 'wsl command'`
- rsync uses `--rsync-path "wsl rsync"` to run rsync within WSL
- tmux sessions started via: `wsl tmux new-session ...`
- Conda activated via: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate ml`

---

## What's Been Completed ✅

### 1. Code Refactoring
- ✅ Created `Kfold_HelperFcns.py` - Shared K-fold cross-validation functions
- ✅ Created `run_hyperparam_training.py` - Standalone training script for remote execution
- ✅ Updated notebook to import from shared module (no code duplication)
- ✅ Created `RemoteTempFiles/` directory for synced results

### 2. Notebook Structure
- ✅ Added `REMOTE_EXECUTION = True/False` flag (cell 3)
- ✅ Added remote helper functions (sync_to_remote, start_remote_training, etc.)
- ✅ Modified training cell (cell 11) with `if REMOTE_EXECUTION:` conditional
- ✅ Added monitoring cells (check_remote_status, tail_remote_log, sync_results_back, load_remote_results)

### 3. Remote Configuration
- Remote host: `cgw35@desktop-win` (desktop-k1molsb)
- Conda environment: `ml` (located at `/home/cgw35/miniconda3/envs/ml`)
- Remote path: `~/projects/TESS_Cluster_Age_ML` (WSL path)
- Tmux session name: `hyperparam_training`

### 4. SSH Authentication
- ✅ SSH key authentication configured and working
- ✅ Public key placed in `C:\ProgramData\ssh\administrators_authorized_keys` (Windows admin location)
- ✅ Password-free SSH connections now functional
- ✅ Automated subprocess calls work without password prompts

### 5. WSL Integration
- ✅ All commands wrapped in `wsl` prefix to run in WSL environment
- ✅ Using existing `~/projects` directory in WSL
- ✅ rsync configured with `--rsync-path "wsl rsync"`
- ✅ Conda activation via `source ~/miniconda3/etc/profile.d/conda.sh`
- ✅ tmux runs within WSL environment

## Current Status: READY TO TEST! 🚀

All blockers have been resolved:
- ✅ SSH key authentication working (admin authorized_keys location fixed)
- ✅ WSL commands accessible via `wsl` prefix
- ✅ Conda environment `ml` confirmed to exist
- ✅ Remote paths updated to use existing `~/projects` directory
- ✅ All remote execution code updated with WSL wrappers

## Next Steps 🔄

### 1. Test Remote Workflow (Ready to test!)

Now test the complete remote training workflow with WSL integration:

#### Step 1: Verify WSL Access (from Mac)
```bash
# Test basic WSL command
ssh cgw35@desktop-win 'wsl pwd'

# Should show: /home/cgw35 or /mnt/c/Users/cgw35

# Verify conda environment
ssh cgw35@desktop-win 'wsl ~/miniconda3/bin/conda env list'

# Should list 'ml' environment
```

#### Step 2: Test Manual rsync to WSL
```bash
# Sync a small test file first
ssh cgw35@desktop-win 'wsl mkdir -p ~/projects'

# Then sync full repo
rsync -avz --exclude 'TempFiles/' --exclude '.git/' \
      --exclude 'RemoteTempFiles' \
      --rsync-path "wsl rsync" \
      ~/Desktop/TESS_Cluster_Age_ML/ \
      cgw35@desktop-win:~/projects/TESS_Cluster_Age_ML/

# Verify it arrived
ssh cgw35@desktop-win 'wsl ls -la ~/projects/TESS_Cluster_Age_ML'
```

#### Step 3: Test Remote Training from Notebook
1. Open `1_HyperparamTuning.ipynb`
2. Set `REMOTE_EXECUTION = True` (already set)
3. Run the remote execution cell
4. Should sync repo and start tmux session automatically
5. Should complete without password prompts

#### Step 4: Monitor Training
Use the monitoring cells in the notebook:
- `check_remote_status()` - Check if tmux session is running
- `tail_remote_log(lines=100)` - View training logs in real-time
- `sync_results_back()` - Copy results from remote to local
- `load_remote_results()` - Load and analyze remote results

#### Manual Debugging (if needed)
```bash
# SSH and enter WSL
ssh cgw35@desktop-win
wsl

# Check tmux sessions
tmux ls

# Attach to training session
tmux attach -t hyperparam_training

# Detach: Ctrl+b, then d
```

### 2. Known Considerations
- **First sync might be slow**: Full repo sync ~348 light curves + code
- **WSL overhead**: Slight performance overhead from running through WSL layer
- **Tmux persistence**: Sessions survive SSH disconnects (good for long training)
- **Conda paths**: Must use full path or source conda.sh before activation

### 3. Potential Issues to Watch For
- rsync might fail if `wsl rsync` not found → install via `sudo apt install rsync` in WSL
- tmux might not exist → install via `sudo apt install tmux` in WSL
- Conda initialization might fail → check `~/miniconda3/etc/profile.d/conda.sh` exists

## Files Created

```
GaussianNN_CGW/Model_Pipeline/
├── Kfold_HelperFcns.py              # Shared K-fold functions
├── run_hyperparam_training.py       # Standalone remote training script
├── RemoteTempFiles/                 # Directory for synced results
├── 1_HyperparamTuning.ipynb         # Updated with remote execution
└── 1_HyperparamTuning_BACKUP_*.ipynb # Backups
```

## Workflow Once Working

### Local Training (Current/Default):
```python
REMOTE_EXECUTION = False
# Run cells normally
```

### Remote Training:
```python
REMOTE_EXECUTION = True
# Run cells → syncs repo, starts training on desktop-win
# Training runs in tmux session (non-blocking)
# Use monitoring cells to check progress
# sync_results_back() when done
# load_remote_results() to analyze
```

## Architecture

```
Local Mac (Jupyter)
    ↓ rsync repo
desktop-win:~/workspace/TESS_Cluster_Age_ML/
    ↓ tmux session: "hyperparam_training"
    ↓ conda activate ml
    ↓ python run_hyperparam_training.py
    ↓ saves: TempFiles/ModelGridLogs/training_results_TIMESTAMP.pkl
    ↓ logs:  TempFiles/logs/training_TIMESTAMP.log
    ↑ rsync results back
Local Mac: RemoteTempFiles/
```

## Decision Made ✅

**SSH Key Authentication (Option A) - IMPLEMENTED**
- ✅ Most secure approach
- ✅ Best practice for automated workflows
- ✅ No passwords stored anywhere
- ✅ Works seamlessly with subprocess calls
- ✅ Solution: Use `C:\ProgramData\ssh\administrators_authorized_keys` for Windows admin accounts

The remote training workflow should now function completely automated without any password prompts!
