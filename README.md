# Fault-Tolerant Distributed Training (Checkpoint / Restore Demo)

This project is a minimal, end-to-end demonstration of **fault-tolerant machine-learning training**, simulating how a job can:

1. **Start on Machine A**  
2. **Get pre-empted** (spot instance termination / SIGTERM)  
3. **Save a safe checkpoint before shutting down**  
4. **Migrate the checkpoint to Machine B**  
5. **Resume training from the exact batch it left off**

It uses PyTorch, MNIST, and a lightweight orchestrator script that emulates real-world behaviour seen in pre-emptible HPC / distributed GPU environments.

---

## ✨ Features

### 🔹 Checkpointing & Atomic Saves
- Saves model weights, optimizer state, epoch, batch index, and loss.  
- Atomic write via `.tmp → os.replace()` to prevent corruption.  
- Debug metadata stored in `meta.json`.

### 🔹 Graceful Shutdown (SIGTERM / SIGINT)
`train.py` installs a custom handler that:
- Catches SIGTERM from the orchestrator  
- Saves a final checkpoint at the next safe point  
- Exits cleanly

This simulates spot instance termination.

### 🔹 Fault-Tolerant Resume
Using `--resume`, the script:
- Restores states  
- Skips completed batches mid-epoch  
- Continues seamlessly from the last checkpoint

### 🔹 Multi-Machine Orchestration
`orchestrator.py`:
- Launches training on *Machine A*  
- Waits ~15 seconds  
- Sends SIGTERM to simulate a pre-emption  
- Copies checkpoints → *Machine B*  
- Restarts training from the same batch

### 🔹 Optional Preemption Simulation
Trigger a fake pre-emption after N batches:
```bash
python train.py --simulate-preempt 300
