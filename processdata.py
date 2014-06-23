# coding: utf-8
import os
import csv
import math
import logging
from collections import namedtuple
from operator import attrgetter

from py2neo import neo4j
from redis import StrictRedis
import networkx as nx
import matplotlib
import requests


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


matplotlib.use('Agg')


# Connections

redis = StrictRedis(host='localhost', port='6379', db=0)
neo = neo4j.GraphDatabaseService("http://138.91.93.45:7474/db/data/")


# Helpers

def chunked(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        if i + n <= len(l):
            yield l[i:i + n]


# Points

WEIGHTS_FIVE = [5.0 / 9.0, 10.0 / 9.0, 15.0 / 9.0, 10.0 / 9.0, 5.0 / 9.0]
WEIGHTS_FIVE_MH = [5.0 / 18.0, 15.0 / 18.0, 50.0 / 18.0, 15.0 / 18.0, 5.0 / 18.0]


Point = namedtuple('Point', 'x y z ts')


def load_point(key):
    return Point(redis.hgetall(key))


def store_point(key, point):
    redis.hmset(key, {
        'x': point.x,
        'y': point.y,
        'z': point.z,
        'ts': point.ts
    })


def wavg(dimension, weights, points):
    if len(weights) != len(points):
        raise ValueError("Weights must be the same length as points")
    values = [getattr(p, dimension) for p in points]
    weighted = [w * v for w, v in zip(weights, values)]
    return float(sum(weighted)) / float(len(weighted))


def avg(dimension, points):
    return wavg(dimension, [1] * len(points), points)


def mean(points):
    return (
        avg('x', points),
        avg('y', points),
        avg('z', points)
    )


def weighted(weights, points):
    return (
        wavg('x', weights, points),
        wavg('y', weights, points),
        wavg('z', weights, points)
    )


def weighted3(points):
    return weighted([0.75, 1.5, 0.75], points)


def midheavy3(points):
    return weighted([0.5, 2, 0.5], points)


# Pending

def get_sample_length(csv_fname):
    """
    how many do i have in the list. Use file name like "accel:raw:0621:235050"
    """
    return len(redis.keys(csv_fname + '*'))


def create_base_for_processing(csv_fname):
    """
    Create a list to process later on. Use file name like "accel:raw:0621:235050"
    """
    # get the length of this dataset
    set_len = get_sample_length(csv_fname)
    logger.debug('create_base_for_processing _ set length: %s', set_len)

    test_str = csv_fname.split(':')

    try:
        if test_str[4] == '':
            pass
    except:
        csv_fname = csv_fname + ':'

    data = []

    for i in range(1, set_len + 1):
        p = load_point(csv_fname + str(i))
        data.append(p)

    return data





def get_set_stat(data):
    """
    this gets the mean, stddev and variance for given list
    """
    sumamt = 0.0

    for i in range(0, len(data)):
        sumamt += data[i]

    avg = sumamt / len(data)

    sumamt = 0.0

    for i in range(0, len(data)):
        sumamt += math.pow((data[i] - avg), 2)

    var = sumamt / len(data)
    stddev = math.sqrt(var)

    out = [avg, stddev, var]
    return out


def downsample(raw_data, averager, downsamplesize=5, csv_fname=None):
    """
    downsamples the data with the given average method

    avg_method can be mean, weight or weight_mh
    """
    if downsamplesize not in (3, 5):
        raise Exception("Only downsample by 3 or 5")

    test_str = csv_fname.split(':')
    try:
        if test_str[4] == '':
            pass
    except:
        csv_fname = csv_fname + ':'

    output = []
    for i, points in enumerate(chunked(raw_data, downsamplesize)):
        mid = points[int(downsamplesize / 2)]
        key = test_str[0] + ':proc:' + test_str[2] + ':' + test_str[3] + ':' + str(i + 1)
        ax, _, az = averager(points)

        p = Point(x=ax, y=0, z=az, ts=mid.ts)
        store_point(key, p)
        output.append(p)

    return output


def sub_avg(ds_data):
    """
    Get a list of ALREADY DOWNSAMPLED accel_datapoint objects, return the same object with averages taken out of X and Z directions
    This is not submitted to redis as it's an easy function to run

    Output is [ [x_avg,x_stddev,x_var], [z_avg,z_stddev,z_var], accel_datapoint list]
    """
    x_list = []
    z_list = []
    for i in range(0, len(ds_data)):
        x_list.append(ds_data[i].accelX)
        z_list.append(ds_data[i].accelZ)

    x_stat = get_set_stat(x_list)
    z_stat = get_set_stat(z_list)

    out = []

    for i in range(0, len(ds_data)):
        test_str = ds_data[i].keyname.split(':')
        p = Point(ts=ds_data[i].timestamp, x=ds_data[i].accelX - x_stat[0], y=0, z=ds_data[i].accelZ - z_stat[0], key=test_str[0] + ':ds-avg:' + test_str[2] + ':' + test_str[3] + ':' + str(i + 1))
        out.append(p)

    return [x_stat, z_stat, out]


def step_fnc(ds_avg_dataset, x_stats, z_stats, stddev_w=1.0):
    """
    uses basic statistical data to turn the downsamples (avg's taken out) into step functions
    the Y axis is used as a placeholder for the cross correleation
    """
    length = len(ds_avg_data)

    out = []
    for i in range(0, length):
        if ds_avg_data[i].accelX < x_stat[0] - stddev_w * x_stat[1]:
            temp_x = -0.5

        elif ds_avg_data[i].accelX > x_stat[0] + stddev_w * x_stat[1]:
            temp_x = 0.5

        else:
            temp_x = 0.0

        if ds_avg_data[i].accelZ < x_stat[0] - stddev_w * x_stat[1]:
            temp_z = -0.5

        elif ds_avg_data[i].accelZ > x_stat[0] + stddev_w * x_stat[1]:
            temp_z = 0.5

        else:
            temp_z = 0.0

        test_str = ds_avg_data[i].keyname.split(':')
        p = Point(ts=ds_avg_dataset[i].timestamp, x=temp_x, y=abs(temp_x) + abs(temp_z), z=temp_z, key=test_str[0] + ':step:' + test_str[2] + ':' + test_str[3] + ':' + str(i + 1))
        store_point(p)

        out.append(p)

    return out


def dump_to_csv(filename, dataset):
    """
    dump any given dataset into a .csv file
    """
    length = len(dataset)
    f = open(filename + '.csv', 'wb')
    try:
        writer = csv.writer(f)
        writer.writerow(('Timestamp', 'AccelX', 'AccelY', 'AccelZ'))

        for i in range(0, len(dataset)):
            writer.writerow((str(dataset[i].timestamp), str(dataset[i].accelX), str(dataset[i].accelY), str(dataset[i].accelZ)))

    except:
        return -1
    finally:
        f.close()

    return 1


def discover_turns(dataset):
    """
    this function takes a accel_datapoint list, which is assumed to be already converted to step function
    It discoveres every cross correleation with amplitude of 1 and records the movement

    an example data output is

    {'start': time, 'end': time, 'turn_begin': [time_list], 'turn_end' : [time_list], 'turn_count' = (count(turns)) }
    """

    start_time = dataset[0].timestamp
    end_time = dataset[-1].timestamp

    turn_begin = []
    turn_end = []

    turn_count = 0

    i = 0
    lastzero = 0
    turn_flag = 0
    while i < len(dataset):

        if dataset[i].accelY == 1.0:
            turn_flag = 1
            logger.debug("lastzero: %s", lastzero)
            i = lastzero
            continue

        elif dataset[i].accelY == 0.0:
            lastzero = i

        if turn_flag == 1:
            j = 1  # The reason you start from one is not to infinite loop on "lastzero" node

            turn_begin.append(dataset[i].timestamp)
            turn_count += 1

            # search until you see next zero that signals the end of the turn

            while(True):
                if dataset[i + j].accelY == 0.0:
                    turn_end.append(dataset[i + j].timestamp)
                    break
                logger.debug("j: %s", j)
                j += 1

            i += j
            turn_flag = 0
            continue

        logger.debug("i: %s", i)
        i += 1

    out = {'start': start_time, 'end': end_time, 'turn_begin': turn_begin, 'turn_end': turn_end, 'turn_count': turn_count}
    return out


def linkdata(turndata):
    """
    this function takes a turn_data dictionary and converts it to link data. The network topology is known (BIG ASS ASSUMPTION)

    turndata = {'start' : start_time, 'end' : end_time, 'turn_begin' : turn_begin, 'turn_end' : turn_end, 'turn_count': turn_count}
    """
    link1_tt = turndata['turn_begin'][0] - turndata['start']

    link2_tt = turndata['end'] - turndata['turn_end'][0]

    return [link1_tt, link2_tt]


if __name__ == "__main__":
    speed = 'high'
    filename = 'accel:raw:0622:144729'

    test = create_base_for_processing(filename)

    processed_data = downsample(raw_data=test, csv_fname=filename)

    returned = sub_avg(processed_data)

    x_stat = returned[0]
    z_stat = returned[1]

    ds_avg_data = returned[2]

    step_out = step_fnc(ds_avg_dataset=ds_avg_data, x_stats=x_stat, z_stats=z_stat, stddev_w=1.5)

    turntest = discover_turns(step_out)

    tt = linkdata(turntest)

    # START NEO TEST

    # STATIC PATH NODES A-C-E-F
    # DYNAMIC PATH NODES A-B-D-F

    for i in neo.find("ROADNODE", property_key="nID", property_value="A"):
        nod_a = i
    for i in neo.find("ROADNODE", property_key="nID", property_value="B"):
        nod_b = i
    for i in neo.find("ROADNODE", property_key="nID", property_value="C"):
        nod_c = i
    for i in neo.find("ROADNODE", property_key="nID", property_value="D"):
        nod_d = i
    for i in neo.find("ROADNODE", property_key="nID", property_value="E"):
        nod_e = i
    for i in neo.find("ROADNODE", property_key="nID", property_value="F"):
        nod_f = i

    link_ab = neo.match_one(start_node=nod_a, end_node=nod_b)
    link_bd = neo.match_one(start_node=nod_b, end_node=nod_d)
    link_df = neo.match_one(start_node=nod_d, end_node=nod_f)

    link_ac = neo.match_one(start_node=nod_a, end_node=nod_c)
    link_ce = neo.match_one(start_node=nod_c, end_node=nod_e)
    link_ef = neo.match_one(start_node=nod_e, end_node=nod_f)

    link_bd["tt"] = tt[0]
    link_df["tt"] = tt[1]

    # SHORTEST PATH

    payload = """{"to" : "http://localhost:7474/db/data/node/<START>", "cost_property" : "tt", "relationships" : {"type" : "CONNECTS_TO","direction" : "out"},"algorithm" : "dijkstra"}"""

    reply = requests.post('http://localhost:7474/db/data/node/0/paths', data=payload)

    # GRAPH

    G = nx.Graph()
    pos = {
        0: (0, 0),
        1: (1, 1),
        2: (1, -1),
        3: (2, 1),
        4: (2, -1),
        5: (3, 0),
    }

    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(1, 3)
    G.add_edge(2, 4)
    G.add_edge(3, 5)
    G.add_edge(4, 5)

    G = G.to_directed()

    G.add_path([0, 1, 3, 5])
    G.add_path([0, 2, 4, 5])

    nx.draw_networkx_nodes(G, pos, node_size=400, nodelist=[0, 5], node_color='#FF0000')
    nx.draw_networkx_nodes(G, pos, node_size=300, nodelist=[1, 2, 3, 4], node_color='#00FFFF')

    # print G.edges()

    # DECIDE LEFT OR RIGHT
    if reply.json()[0]['nodes'][1].split("/")[6] == unicode('1'):
        print "sol kisa"
        nx.draw_networkx_edges(G, pos, width=3, edgelist=[(0, 1)], edge_color='#00CC00', style='dashed', arrows=True)
        nx.draw_networkx_edges(G, pos, width=3, edgelist=[(1, 3), (3, 5)], edge_color='#00CC00', arrows=True)
        nx.draw_networkx_edges(G, pos, width=1, edgelist=[(0, 2)], edge_color='#000000', style='dashed', arrows=True)
        nx.draw_networkx_edges(G, pos, width=1, edgelist=[(2, 4), (4, 5)], edge_color='#000000', arrows=True)
    else:
        print "sag kisa"
        nx.draw_networkx_edges(G, pos, width=1, edgelist=[(0, 1)], edge_color='#000000', style='dashed', arrows=True)
        nx.draw_networkx_edges(G, pos, width=1, edgelist=[(1, 3), (3, 5)], edge_color='#000000', arrows=True)
        nx.draw_networkx_edges(G, pos, width=3, edgelist=[(0, 2)], edge_color='#00CC00', style='dashed', arrows=True)
        nx.draw_networkx_edges(G, pos, width=3, edgelist=[(2, 4), (4, 5)], edge_color='#00CC00', arrows=True)

    nx.draw_networkx_labels(G, pos, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={
        (0, 1): link_ab["tt"],
        (1, 3): link_bd["tt"],
        (3, 5): link_df["tt"],
        (0, 2): link_ac["tt"],
        (2, 4): link_ce["tt"],
        (4, 5): link_ef["tt"]
    }, label_pos=0.5, font_size=10, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None, rotate=True)

    matplotlib.pyplot.axis('on')
    matplotlib.pyplot.savefig("hackathon.png")
    os.system("cp hackathon.png samplewebserver/static/hackathon.png")
