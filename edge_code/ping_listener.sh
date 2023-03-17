#!/bin/bash

while true
do
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
        python3 ~/Fleet/train.py "$cid" "$strategy" &
    fi
done