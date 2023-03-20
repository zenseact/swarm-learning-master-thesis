#!/bin/bash

# Set the repository URL and local directory for cloning the repository
REPO_URL="https://github.com/your_username/your_repository.git"
LOCAL_DIR="/path/to/local/directory"

# Function to check for and pull updates from the repository
check_for_updates() {
    cd "$LOCAL_DIR"
    git fetch origin
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse @{u})
    if [ $LOCAL != $REMOTE ]; then
        git pull origin master
    fi
}

while true
do
    # Check for updates before executing the Python script
    check_for_updates

    # Listen for incoming ping requests
    ping_request=$(sudo tcpdump -i eth0 -c 1 icmp and icmp[icmptype]=icmp-echo)

    # If a ping request was received, extract the message and execute your Python script
    if [[ ! -z "$ping_request" ]]
    then
        # Extract the IP header and message from the ping request
        ip_header=$(echo "$ping_request" | awk -F ' IP ' '{print $2}')
        message=$(echo "$ip_header" | awk -F ':' '{print $1,$2}')

        # Pass the message as two separate parameters to your Python script
        cid=$(echo "$message" | awk '{print $1}')
        strategy=$(echo "$message" | awk '{print $2}')

        # Execute the Python script after checking for updates
        python3 ~/Fleet/train.py "$cid" "$strategy" &
    fi
done
