# SEC_pipeline
An easy-to-use interface to train a Sound-Event Classificatio (SEC) pipeline by means of CLAP.

### Installation and configuration
1. Clone repository:
   `git clone https://github.com/LJPileggi/SEC_pipeline.git`
2. Build sif file containing all the necessary libraries. You can either pull the image locally on your machine
   from its GitHub Container Registry via `apptainer pull --force clap_pipeline.sif ghcr.io/ljpileggi/dockerfile_for_sec_pipeline:latest`
   (requires apptainer plus ~30GB available on your memory to perform the build), or you can use the build_sif_from_image.ipynb
   notebook available in the library to build it remotely and then download it on your local machine of your Google Drive
3. run `./setup_CLAP_env.sh <user_area> <remote_config> <Cineca_Uploads>` a first time (user_area is your area containing the SEC_pipeline
   directory, remote_config is the rclone config required to upload the container from your start endpoint, Cineca_Uploads is the
   remote path containing your sif file). If rclone is not present on your filesystem, it will try to install it. The script tries
   to find the remote_config configuration; if not present, the script is stopped to let you configure it manually. If present,
   proceeds with the container upload from its remote location of your choice
4. run `rclone config` or set `RCLONE_ABS_PATH="/your/own/area/SEC_pipeline/bin/rclone"` as environment variable and then run `RCLONE_ABS_PATH config`
   to setup your remote configuration. If your container is stored on services requiring an OAuth2 authentication method (like
   Google Drive or Dropbox), you have to first exit your current session, configure an SSH tunnel like so `ssh -L <LOCAL_PORT>:<INTERNAL_REMOTE_IP>:<REMOTE_PORT> user@login.supercomputer.it`; once in, run `export PATH="/your/own/area/SEC_pipeline/bin:$PATH"`, then `RCLONE_ABS_PATH="/your/own/area/SEC_pipeline/bin/rclone"` and `"$RCLONE_ABS_PATH" config reconnect <remote_config>:` and authenticate with your account. Then run
   `./setup_CLAP_env.sh <user_area> <remote_config> <Cineca_Uploads>` to complete the sif donwload
