import os
import sys
import time
import subprocess
import signal
import shutil
import platform


MACHINE_A_DIR = "./machine-A"
MACHINE_B_DIR = "./machine-B"     
CHECKPOINT_DIR_NAME = "checkpoints"

MACHINE_A_CHECKPOINTS = os.path.join(MACHINE_A_DIR, CHECKPOINT_DIR_NAME)
MACHINE_B_CHECKPOINTS = os.path.join(MACHINE_B_DIR, CHECKPOINT_DIR_NAME)      # in reality these would be on 2 different AWS EC2 instances 

# runs subprocesses with the same python interpreter
PYTHON_EXE = sys.executable 


def clear_previous_runs():
    """Wipe the directories from the last run to start fresh."""         
    print("[orchestrator] Clearing previous runs...")
    if os.path.exists(MACHINE_A_DIR):
        shutil.rmtree(MACHINE_A_DIR)
    if os.path.exists(MACHINE_B_DIR):
        shutil.rmtree(MACHINE_B_DIR)
    
    os.makedirs(MACHINE_A_CHECKPOINTS, exist_ok=True)
    os.makedirs(MACHINE_B_CHECKPOINTS, exist_ok=True)

def print_header(title):
    """Prints a formatted header."""
    print("\n" + "="*60)
    print(f"    {title}")
    print("="*60 + "\n")

#  Main Orchestration Logic 
def run_orchestration():
    clear_previous_runs()

    
    print_header("STAGE 1: Starting job on Machine-A")
    
    
    cmd_machine_a = [                         
        PYTHON_EXE, "train.py",
        "--checkpoint-dir", MACHINE_A_CHECKPOINTS,
        "--epochs", "5",
        "--save-every", "50",  # Checkpoint every batche
        "--log-every", "50"
    ]
    
    print(f"[orchestrator] Executing: {' '.join(cmd_machine_a)}")


    start_new_session = os.setsid if sys.platform != "win32" else None   # creates a prog group on Unix so we can handle shutdown easier
    
    try:                                  
        job_a = subprocess.Popen(                  
            cmd_machine_a,           
            stdout=sys.stdout,       # for terminal debugging
            stderr=sys.stderr,
            preexec_fn=start_new_session    # create a new process group, which then sets up for the graceful kill 
        )
    except Exception as e:
        print(f"[orchestrator] Failed to start process: {e}")
        return

    # --- STAGE 2: Simulate a preemption event ---
    print_header("STAGE 2: Simulating preemption")
    
    # Let the job run 
    simulate_run_time = 15
    print(f"[orchestrator] Letting job run for {simulate_run_time} seconds...")
    time.sleep(simulate_run_time)
    
    print(f"\n[orchestrator] !!! SPOT PREEMPTION IMMINENT !!!")
    print(f"[orchestrator] Sending SIGTERM to process group {job_a.pid}...")
    
    # Send SIGTERM (this is what GracefulKiller catches)
    if sys.platform != "win32":
        os.killpg(job_a.pid, signal.SIGTERM)     # sends the graceful shutdown request to the train.py and any other child processes 
    else:
        job_a.send_signal(signal.SIGTERM) 

    try:
        job_a.wait(timeout=10) 
        print("[orchestrator] Job on Machine-A shut down gracefully.")
    except subprocess.TimeoutExpired:
        print("[orchestrator] Job did not exit! Forcing SIGKILL.")
        job_a.kill()

    # --- STAGE 3: Migrate the checkpoint ---
    print_header("STAGE 3: Migrating checkpoint from A -> B")
    
    if not os.path.exists(os.path.join(MACHINE_A_CHECKPOINTS, "checkpoint.pth")):
        print("[orchestrator] ERROR: No checkpoint file found on Machine-A!")
        print("[orchestrator] This means the 'train.py' script failed to save.")
        return
        
    print(f"[orchestrator] Copying '{MACHINE_A_CHECKPOINTS}' to '{MACHINE_B_CHECKPOINTS}'")
    
    
    shutil.copytree(MACHINE_A_CHECKPOINTS, MACHINE_B_CHECKPOINTS, dirs_exist_ok=True)
    
    print("[orchestrator] Migration complete.")

    # --- STAGE 4: Resume the job on Machine B ---
    print_header("STAGE 4: Resuming job on Machine-B")
    
    # Note the new checkpoint dir and the --resume flag
    cmd_machine_b = [
        PYTHON_EXE, "train.py",
        "--checkpoint-dir", MACHINE_B_CHECKPOINTS,
        "--epochs", "5",
        "--save-every", "50",
        "--log-every", "50",
        "--resume" # The critical flag
    ]
    
    print(f"[orchestrator] Executing: {' '.join(cmd_machine_b)}")
    
    try:
        subprocess.run(
            cmd_machine_b, 
            stdout=sys.stdout, 
            stderr=sys.stderr, 
            check=True
        )
        print("\n[orchestrator] Job finished successfully on Machine-B.")
    except subprocess.CalledProcessError:
        print("[orchestrator] Resumed job on Machine-B failed!")
    except Exception as e:
        print(f"[orchestrator] Failed to resume process: {e}")

    print_header("ORCHESTRATION SIMULATION COMPLETE")

if __name__ == "__main__":
    try:
        run_orchestration()
    except KeyboardInterrupt:
        print("\n[orchestrator] Caught Ctrl+C, exiting demo cleanly.")