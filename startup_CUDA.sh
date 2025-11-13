#!/bin/bash

ssh ubuntu@<NEW_IP> -i ~/.ssh/gpt2    # CHANGE IP 
cd ~
git clone git@github.com:Marques-079/MLA-Transformer.git
cd MLA-Transformer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_CUDA.txt
python MHA/main_124.py


: '
#local /Users/marcus/Documents/GitHub/MLA-Transformer SIDE nano ~/.ssh/config
Host lambda-gpu
    HostName 209.20.159.97
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes

Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes

#UBUNTU@<IP> SIDE nano ~/.ssh/config
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
'


