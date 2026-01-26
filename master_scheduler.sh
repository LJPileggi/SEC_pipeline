#!/bin/bash
# ==============================================================================
# MASTER CAMPAIGN LAUNCHER - PRODUCTION VERSION (24h)
# ==============================================================================
# This script manages the sequential execution of your embedding campaign.
# It runs on a compute node but requests ZERO GPUs to minimize cost and impact.
# ==============================================================================

#SBATCH --job-name=MASTER_EMB
#SBATCH --partition=boost_usr_prod   # Main production partition
#SBATCH --ntasks=1                   # Only one task for the scheduler
#SBATCH --cpus-per-task=4            # Minimal CPU requirement
#SBATCH --mem=8G                     # Minimal RAM for Python management
#SBATCH --gres=gpu:0                 # 🎯 MANDATORY: Request ZERO GPUs for the Master
#SBATCH --time=24:00:00              # Maximum limit for Leonardo production
#SBATCH --account=IscrC_Pb-skite
#SBATCH --output=master_campaign_%j.out

# 🎯 ENSURE SEQUENTIALITY:
# The scheduler uses 'sbatch --wait', ensuring that this Master job 
# stays blocked until the current worker finishes.

DIRECTIVES_FILE="scheduler_directives/directives.txt"
SCHEDULER_SCRIPT="embedding_pipeline_scheduler.sh"

# Path validation
if [ ! -f "$DIRECTIVES_FILE" ]; then
    echo "❌ Error: Directives file '$DIRECTIVES_FILE' not found."
    exit 1
fi

# Launch the scheduler in blocking mode
# The Master will wait here for each GPU job to complete.
bash "$SCHEDULER_SCRIPT" "$DIRECTIVES_FILE"
