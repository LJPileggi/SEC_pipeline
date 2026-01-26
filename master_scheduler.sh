#!/bin/bash
# ==============================================================================
# MASTER CAMPAIGN LAUNCHER
# ==============================================================================
# This script acts as a persistent supervisor for the entire embedding campaign.
# It runs in the 'serial' partition to avoid using GPU resources while waiting 
# for the sequential worker jobs to finish.
#
# Use: sbatch master_launcher.sh
# ==============================================================================

#SBATCH --job-name=MASTER_EMB
#SBATCH --partition=serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00           # Maximum duration for the entire campaign
#SBATCH --account=IscrC_Pb-skite  # Your Cineca Account
#SBATCH --output=master_campaign_%j.out

# 🎯 CONFIGURATION
# Name of the file containing the list of (config|format|octave) tasks
DIRECTIVES_FILE="scheduler_directives/directives.txt"
# The main scheduler script that handles 'sbatch --wait' logic
SCHEDULER_SCRIPT="embedding_pipeline_scheduler.sh"

# 🔍 VALIDATION
if [ ! -f "$DIRECTIVES_FILE" ]; then
    echo "❌ CRITICAL: Directives file '$DIRECTIVES_FILE' not found."
    exit 1
fi

if [ ! -f "$SCHEDULER_SCRIPT" ]; then
    echo "❌ CRITICAL: Scheduler script '$SCHEDULER_SCRIPT' not found."
    exit 1
fi

# 🚀 EXECUTION
echo "📖 [MASTER] Starting Campaign Supervisor at $(date)"
echo "🔗 [MASTER] Directives: $DIRECTIVES_FILE"

# Execute the scheduler. The scheduler uses 'sbatch --wait', so this process 
# will remain active and blocked until the last GPU job in the directives finishes.
#
bash "$SCHEDULER_SCRIPT" "$DIRECTIVES_FILE"

echo "🏁 [MASTER] Campaign Finished at $(date)"
