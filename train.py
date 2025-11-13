import argparse
import os
import time
import signal
import sys
import json
import math
import random
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --- Constants ---
CHECKPOINT_FILE = "checkpoint.pth"      # where the model state is saved
META_FILE = "meta.json"         # thiss is just extra debugging info about checkpoints


# -------------------------- Utils --------------------------

def set_seed(seed: int = 1337):       # this is just so we get the same results everytime we run the code , as ml has lots of randomness
    random.seed(seed)
    torch.manual_seed(seed)     # for pytorchs cpu random number gen
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):                 # this func bascially just creates the parent folders we need
    os.makedirs(path, exist_ok=True)


def save_meta(checkpoint_dir: str, meta: Dict[str, Any]):             # this function saves metadata like why a checkpoint was saved or how many batches have been processed into  a .json file 
    """Save metadata about the checkpoint (for debugging/monitoring)."""
    meta_path = os.path.join(checkpoint_dir, META_FILE)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_meta(checkpoint_dir: str) -> Dict[str, Any]:
    """Load checkpoint metadata if it exists."""
    meta_path = os.path.join(checkpoint_dir, META_FILE)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


# -------------------------- Model --------------------------

class Net(nn.Module):
    """
    Simple CNN for MNIST.
    Architecture evolved slightly from v1 - added dropout for better generalization.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# -------------------------- Checkpointing --------------------------

def save_checkpoint(checkpoint_dir: str,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    batch_idx: int,
                    loss: float,
                    best_val_loss: float = math.inf,
                    reason: str = "periodic"):
  
    ensure_dir(checkpoint_dir)
    
    print(f"=> Saving checkpoint at Epoch {epoch}, Batch {batch_idx} (reason: {reason})")
    
    state = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_val_loss': best_val_loss,
        'timestamp': time.time(),
        'reason': reason,
    }
    
    # Atomic save to avoid corruption
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILE)
    temp_file = checkpoint_path + ".tmp"
    torch.save(state, temp_file)
    os.replace(temp_file, checkpoint_path)


def load_checkpoint(checkpoint_dir: str,
                    model: nn.Module,
                    optimizer: optim.Optimizer) -> Tuple[int, int, float]:

    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILE)
    
    if not os.path.exists(checkpoint_path):
        print("=> No checkpoint found. Starting training from scratch.")
        return 0, 0, math.inf

    print(f"=> Loading checkpoint from '{checkpoint_path}'...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    start_batch_idx = checkpoint['batch_idx']
    loss = checkpoint['loss']
    best_val_loss = checkpoint.get('best_val_loss', math.inf)
    reason = checkpoint.get('reason', 'unknown')
    
    print(f"=> Loaded checkpoint. Resuming from Epoch {start_epoch}, Batch {start_batch_idx}")
    print(f"   Last saved loss: {loss:.4f} | Best val loss: {best_val_loss:.4f} | Reason: {reason}")
    
    return start_epoch, start_batch_idx, best_val_loss


# -------------------------- Signal Handling --------------------------

class GracefulKiller:
    
    def __init__(self):
        self.triggered = False
        # Register handlers for both Ctrl+C (SIGINT) and orchestrator kills (SIGTERM)
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self.triggered = True
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n[signal] Caught {sig_name}. Will checkpoint and exit at the next safe point...")


# -------------------------- Evaluation --------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device):
    """
    NEW in v2: Added validation to track best model.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * targets.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# -------------------------- Training Loop --------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ensure_dir(args.checkpoint_dir)
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"\nFailed to download MNIST: {e}")
        return
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    num_batches = len(trainloader)
    
    # Setup model and optimizer
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Resume logic
    start_epoch = 0
    start_batch_idx = 0
    best_val_loss = math.inf
    
    if args.resume:
        try:
            start_epoch, start_batch_idx, best_val_loss = load_checkpoint(
                args.checkpoint_dir, model, optimizer
            )
        except FileNotFoundError:
            print("Checkpoint file not found. Starting from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            checkpoint_path = os.path.join(args.checkpoint_dir, CHECKPOINT_FILE)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
    else:
        print("[start] Fresh run (no --resume flag)")
    
    # signal handler for graceful shutdown
    killer = GracefulKiller()
    
    # track total batches processed across all epochs
    processed_batches_global = start_epoch * num_batches + start_batch_idx
    
    # optional simulated preemption for testing
    def maybe_simulate_preempt():
      
        if args.simulate_preempt is not None and processed_batches_global >= args.simulate_preempt:
            print(f"[simulate] Reached {processed_batches_global} batches; simulating preemption.")
            # Save current state
            next_batch = batch_idx + 1
            next_epoch, next_idx = (epoch + 1, 0) if next_batch >= num_batches else (epoch, next_batch)
            save_checkpoint(
                args.checkpoint_dir, model, optimizer, next_epoch, next_idx,
                loss.item(), best_val_loss, reason="simulated_preemption"
            )
            save_meta(args.checkpoint_dir, {
                "last_reason": "simulated_preemption",
                "processed_batches": processed_batches_global
            })
            print("[exit] Simulated preemption - exiting.")
            sys.exit(2)
    
    # Training loop
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        
        data_iterator = iter(trainloader)           # if resuming mid-epoch, we need to skip batches we already processed

        
        if start_batch_idx > 0:
            print(f"[resume] Skipping to batch {start_batch_idx} in epoch {epoch}...")
            for _ in range(start_batch_idx):
                try:
                    next(data_iterator)
                except StopIteration:
                    break
        
        # Process batches starting from start_batch_idx
        for batch_idx in range(start_batch_idx, num_batches):
            data, targets = next(data_iterator)
            data, targets = data.to(device), targets.to(device)
            
            # Standard training step
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            processed_batches_global += 1
            
            # Logging
            if (batch_idx + 1) % args.log_every == 0 or batch_idx == 0:
                print(f"Epoch: {epoch}/{args.epochs-1} | Batch: {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")










            
            # periodic checkpointing 
            if args.save_every and processed_batches_global % args.save_every == 0:
                next_batch = batch_idx + 1
                next_epoch, next_idx = (epoch + 1, 0) if next_batch >= num_batches else (epoch, next_batch)
                save_checkpoint(
                    args.checkpoint_dir, model, optimizer, next_epoch, next_idx,     #model weights, learining rates
                    loss.item(), best_val_loss, reason="periodic"
                )
                save_meta(args.checkpoint_dir, {
                    "last_reason": "periodic",
                    "processed_batches": processed_batches_global
                })
            
            # graceful shutdown on signal
            if killer.triggered:
                next_batch = batch_idx + 1
                next_epoch, next_idx = (epoch + 1, 0) if next_batch >= num_batches else (epoch, next_batch)
                save_checkpoint(
                    args.checkpoint_dir, model, optimizer, next_epoch, next_idx,   
                    loss.item(), best_val_loss, reason="signal"
                )
                save_meta(args.checkpoint_dir, {
                    "last_reason": "signal",
                    "processed_batches": processed_batches_global
                })
                print("[exit] Stopping after graceful checkpoint.")
                return
            





            
            # heck for simulated preemption
            maybe_simulate_preempt()
        
        # End of epoch - reset start_batch_idx for next epoch
        start_batch_idx = 0
        
        # validation and best model tracking
        val_loss, val_acc = evaluate(model, testloader, device)
        print(f"[epoch {epoch}] val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.checkpoint_dir, model, optimizer, epoch + 1, 0,
                loss.item(), best_val_loss, reason="best_model"
            )
            print(f"[best] New best validation loss: {val_loss:.4f} — checkpoint saved.")
    
    print("[done] Training complete.")


# -------------------------- CLI --------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fault-tolerant MNIST training with checkpoint/restore"
    )
    
    # Training params
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for reproducibility")
    
    # Checkpointing params
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N processed batches (0 to disable periodic saves)")
    
    # Logging and testing
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log training progress every N batches")
    parser.add_argument("--simulate-preempt", type=int, default=None,
                        help="NEW: Simulate spot preemption after N total batches (for testing)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        train(args)
    except KeyboardInterrupt:
        # Fallback handler (GracefulKiller should catch this, but just in case)
        print("\n\nTraining interrupted by user (Ctrl+C).")
        print("The last saved checkpoint can be used to resume with --resume flag.")