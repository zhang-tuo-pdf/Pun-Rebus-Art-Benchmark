#!/bin/bash
set -x  # Enable debugging output

# Download the file
wget -O Pun_Chinese_Painting.zip 'https://www.dropbox.com/scl/fi/1s958fi0f4j8l5nmtf3y5/Pun_Chinese_Painting.zip?rlkey=q9fotvuqt6zvvdni7fc3cbn9u&st=l0i2kzaf&dl=0'

# Check if wget was successful
if [ $? -eq 0 ]; then
    echo "Download successful, attempting to unzip."
    unzip Pun_Chinese_Painting.zip
else
    echo "Download failed, not attempting to unzip."
fi