#!/bin/bash
# Dump DB

if [ -z $1 ]
then
    read -p "Enter the name of the database dump file: "  FILENAME
else
    FILENAME=$1
fi
PGPASSWORD="connectomics" pg_dump -U cluttered_nist -h localhost cluttered_nist > db_dumps/$FILENAME.dump

