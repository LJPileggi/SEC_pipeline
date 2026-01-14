# **SEC\_pipeline**

An easy-to-use interface to train a Sound-Event Classification (SEC) pipeline by means of CLAP.

### **Installation and configuration**

1. Clone repository:  
   git clone https://github.com/LJPileggi/SEC\_pipeline.git  
2. Build sif file containing all the necessary libraries. You can either pull the image locally on your machine from its GitHub Container Registry via apptainer pull \--force clap\_pipeline.sif ghcr.io/ljpileggi/dockerfile\_for\_sec\_pipeline:latest (requires apptainer plus \~30GB available on your memory to perform the build), or you can use the build\_sif\_from\_image.ipynb notebook available in the library to build it remotely and then download it on your local machine of your Google Drive  
3. run ./setup\_CLAP\_env.sh \<user\_area\> \<remote\_config\> \<Cineca\_Uploads\> a first time (user\_area is your area containing the SEC\_pipeline directory, remote\_config is the rclone config required to upload the container from your start endpoint, Cineca\_Uploads is the remote path containing your sif file). If rclone is not present on your filesystem, it will try to install it. The script tries to find the remote\_config configuration; if not present, the script is stopped to let you configure it manually. If present, proceeds with the container upload from its remote location of your choice  
4. run rclone config or set RCLONE\_ABS\_PATH="/your/own/area/SEC\_pipeline/bin/rclone" as environment variable and then run RCLONE\_ABS\_PATH config to setup your remote configuration. If your container is stored on services requiring an OAuth2 authentication method (like Google Drive or Dropbox), you have to first exit your current session, configure an SSH tunnel like so ssh \-L \<LOCAL\_PORT\>:\<INTERNAL\_REMOTE\_IP\>:\<REMOTE\_PORT\> user@login.supercomputer.it; once in, run export PATH="/your/own/area/SEC\_pipeline/bin:$PATH", then RCLONE\_ABS\_PATH="/your/own/area/SEC\_pipeline/bin/rclone" and "$RCLONE\_ABS\_PATH" config reconnect \<remote\_config\>: and authenticate with your account. Then run ./setup\_CLAP\_env.sh \<user\_area\> \<remote\_config\> \<Cineca\_Uploads\> to complete the sif download

### ---

**Production & Embedding Pipeline**

The embedding pipeline is a high-performance system designed to generate audio features (CLAP embeddings and n-octave band spectrograms) on distributed multi-GPU clusters. It is specifically optimized for the Leonardo Cluster, ensuring stable execution on compute nodes even without internet access.

#### **1\. Manual Orchestrator (run\_pipeline.sh)**

The run\_pipeline.sh script is the entry point for manual or interactive operations. It replicates the production environment by automating the setup of local model weights, firewall-safe environment variables, and isolated job-specific workspaces to prevent resource contention.  
**Usage**:

Bash

./run\_pipeline.sh \<config\_file\> \<audio\_format\> \<n\_octave\> \<mode\>

* **config\_file**: Name of the YAML configuration file located in the configs/ directory.  
* **audio\_format**: The extension of the audio files to process (wav, mp3, or flac).  
* **n\_octave**: The desired octave band resolution for the generated spectrograms.  
* **mode**: Use interactive for immediate execution on an already allocated compute node, or slurm to dispatch a single background job to the cluster queue.

#### **2\. Campaign Scheduler (super\_scheduler.sh)**

For large-scale production involving multiple configurations, the super\_scheduler.sh script automates the deployment of job chains. It parses a directives file to launch sequential or parallel tasks, utilizing Slurm's internal dependency system (--dependency=afterok) to ensure a task starts only if the previous one was successful.  
**Usage**:

Bash

./super\_scheduler.sh \<directives\_file\> \<global\_mode\>

* **directives\_file**: Path to the directives text file containing the configuration header and task list.  
* **global\_mode**: Set to sequential (default) to link all tasks via Slurm dependencies, or parallel to submit all tasks at once.

### ---

**Directives Configuration (job\_directives.txt)**

The scheduler requires a structured directives file. This file allows users to define global environment parameters and a list of independent tasks in a single place.  
**Structure and Syntax**:

* **Global Header**: Defines variables such as SIF\_FILE, CLAP\_WEIGHTS, and ROBERTA\_PATH, along with Slurm-specific parameters (Account, Partition, and Walltime).  
* **Execution List**: Defines individual tasks using the format config\_file | audio\_format | n\_octave.

**Example Content**:

Plaintext

\# \==========================================================  
\# GLOBAL SCHEDULER PARAMETERS (HEADER)  
\# \==========================================================  
SIF\_FILE          | /path/to/clap\_pipeline.sif  
CLAP\_WEIGHTS      | /path/to/CLAP\_weights\_2023.pth  
ROBERTA\_PATH      | /path/to/roberta-base  
SLURM\_ACCOUNT     | IscrC\_Pb-skite  
SLURM\_PARTITION   | boost\_usr\_prod  
SLURM\_TIME        | 23:59:59

\# \==========================================================  
\# EXECUTION LIST (Format: config | format | octave)  
\# \==========================================================  
production\_v1.yaml | wav  | 1  
production\_v1.yaml | wav  | 3  
production\_v2.yaml | flac | 1

**Note**: Lines starting with \# are treated as comments and ignored. Fields within the execution list must be separated by a pipe (|).

### ---

**Technical Implementation Details**

* **Firewall Redirection (Offline Mode)**: The pipeline implements a Monkey Patching mechanism to intercept model download requests from msclap and transformers libraries. These requests are redirected to local scratch paths, allowing full operation on firewalled compute nodes.  
* **Isolated Job Workspaces**: To prevent I/O contention and cache corruption, every job (manual or scheduled) creates a unique temporary directory in scratch (tmp\_job\_ID). This workspace stores model weights and tokenizer assets and is automatically deleted upon job completion.  
* **Deterministic Reproducibility**: Audio augmentations, including random offsets and noise, are generated using a deterministic seed system tied to the specific audio record index. This ensures identical results across different runs and hardware.  
* **Resumability and Tracking**: The pipeline tracks progress using log.json at the class level. Workers automatically skip already completed audio classes, enabling seamless job restarts after a timeout or crash.  
* **Environment Consistency**: Through the NODE\_TEMP\_BASE\_DIR variable, the pipeline maps external persistent scratch data to the container's internal filesystem, maintaining compatibility with the dirs\_config.py module.