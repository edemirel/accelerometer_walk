# coding: utf-8
import os
import csv
import urllib2
import contextlib

import redis


CSVSTORE = "http://localhost/static/"


def upload_csv_to_redis(csvfile):
    # explode filename into parts
    file_name = csvfile.replace(".", "_").split("_")

    with contextlib.closing(urllib2.urlopen(csvfile)) as f:
        reader = csv.reader(f)

        # create generic redis line for specific file
        redisline = 'accel:raw:' + str(file_name[1]) + ':' + str(file_name[2]) + ':'

        for r, row in enumerate(reader):
            if r == 0:
                header = row
            else:
                for c, word in enumerate(row):
                    r.hset(redisline + str(r), header[c], word)
                    r.sadd('keys', redisline + str(r))
                    # print redisline + str(rownum)


if __name__ == "__main__":
    r = redis.StrictRedis(host='localhost', port='6379', db=0)

    # I reset the Redis db every run, for no appearent reason
    r.flushdb()

    files = ["%s%s" % (CSVSTORE, f) for f in os.listdir("<DIRECTORY_WHERE_YOU_PUT_YOUR_ACCELEROMETER DATA")]

    for f in files:
        print "Uploading %s" % f
        try:
            upload_csv_to_redis(f)
            print "    ok"
        except Exception as e:
            print "    Ugh... exception!"
