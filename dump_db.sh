#!/bin/bash
# Dump DB

if [ -z $1 ]
then
    read -p "Enter the name of the database dump file: "  FILENAME
else
    FILENAME=$1
fi
PGPASSWORD="connectomics" pg_dump -U connectomics -h localhost connectomics > db_dumps/$FILENAME.dump

