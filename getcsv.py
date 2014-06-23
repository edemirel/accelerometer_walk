# coding: utf-8
import os
import csv
import urllib2

import redis


CSVSTORE = "http://localhost/static/"


def upload_csv_to_redis(csv_filename):
    # explode filename into parts
    file_name = csv_filename.replace(".", "_").split("_")

    csv_f = urllib2.urlopen(CSVSTORE + csv_filename)
    reader = csv.reader(csv_f)

    # open the CSV file from HTML stream
    rownum = 0

    # create generic redis line for specific file
    redisline = 'accel:raw:' + str(file_name[1]) + ':' + str(file_name[2]) + ':'

    for row in reader:
        if rownum == 0:
            header = row

        else:
            colnum = 0
            # prepare redis key for that row

            for word in row:
                r.hset(redisline + str(rownum), header[colnum], word)
                r.sadd('keys', redisline + str(rownum))
                # print redisline + str(rownum)
                colnum += 1

        rownum += 1


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
