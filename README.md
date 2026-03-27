# OVERVIEW
This repository is used for my submission to PrediCT Projects on Google Summer of Code 2026

# GUIDELINE
1. Insert all dataset inside `data_raw` directory
2. *IF USING LINUX* : I am using Linux Ubuntu which blocking to run `pip` directly on host machine. You could use this command to install `venv` (virtual environment) first to run pip on your linux/ubuntu machine if its blocked. 
    ```
    sudo apt update
    sudo apt install python3.12-venv
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install required library using command `pip install -r requirements.txt`