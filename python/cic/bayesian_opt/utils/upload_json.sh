#!/bin/bash

username=$1
password=$2

echo "Upload JSON"
sshpass -p ${password}  scp roboch.json ${username}@robots.real-robot-challenge.com: