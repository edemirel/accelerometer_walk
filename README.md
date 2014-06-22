accelerometer_walk
==================

A simple system that tries to recognizes left/right turns in a network with known topology.

Technologies used: Redis, Neo4j


getcsv.py
==================
Used to get the CSV files in the webserver in the Azure cloud and commit to Redis.


network.csv
==================
The headers are [linkID, from_node, to_node, tt]

tt denotes travel time in seconds

processdata.py
=================
The meat of it. Needs serious scrubbing
