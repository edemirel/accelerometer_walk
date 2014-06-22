from bs4 import BeautifulSoup
import urllib2, csv, redis, os

def upload_csv_to_redis(csv_filename):

	#explode filename into parts
	file_name = csv_filename.replace(".","_").split("_")
	
	csv_f = urllib2.urlopen(CSVSTORE+csv_filename)
	reader = csv.reader(csv_f)

	#open the CSV file from HTML stream
	rownum = 0

	#create generic redis line for specific file
	redisline = 'accel:raw:' + str(file_name[1]) + ':' + str(file_name[2])+ ':'
	
	for row in reader:	
		if rownum == 0:
			header = row

		else:
			colnum = 0
			#prepare redis key for that row

			for word in row:
				r.hset(redisline + str(rownum),header[colnum],word)
				r.sadd('keys',redisline + str(rownum))
				#print redisline + str(rownum)
				colnum += 1

		rownum += 1

if __name__ == "__main__": 

	#CONFIG
	#r = redis.StrictRedis(host='138.91.93.45', port='6379', db=0)
	r = redis.StrictRedis(host='localhost', port='6379', db=0)

	r.flushdb()

	#CLOUDURL = 'http://192.168.0.230:54388/'
	# CLOUDURL = "http://egeneo4j.cloudapp.net/test"
	CSVSTORE = "http://egeneo4j.cloudapp.net/static/"

	# #Read HTML
	# f = urllib2.urlopen(CLOUDURL)	
	# html_file = f.read()

	# # #Convert HTML to BeautifulSoup
	# soup = BeautifulSoup(html_file)

	list_csv = os.listdir("/home/Ege/hackathon/samplewebserver/static/")


	# # #Search for all "a" tags, get href attributes
	# for link in soup.findAll("a"):
	#  	#get attribute in href tag, combine with base link
	#  	#the reason for [8:-1] is dumping the /static/ text
	# 	if(link["href"].find("+Delete") == -1):
	# 		csvlink = CLOUDURL+link["href"]
	# 		list_csv.append(str(link["href"][8:]))

	#print list_csv
	try:
		for i in range(0,len(list_csv)-1):
			upload_csv_to_redis(list_csv[i])
			print 'uploaded ' + list_csv[i]
	except Exception, e:
		print list_csv[i] + " patladi, canin sagolsun"
	


	
