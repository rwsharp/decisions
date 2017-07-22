import argparse

import numpy as np
from datetime import datetime, timedelta
import json

from population import Population


def main(args):
    population = Population.from_json(args.population_file)
    print 'Population size: {}'.format(len(population.people))
    print

    dt_fmt = '%Y%m%d'
    deciles = range(0, 101, 10)

    # Age distribution
    now = datetime.now()

    ages = list()
    for person in population.people.values():
        if person.dob is not None:
            age = now - datetime.strptime(person.dob, dt_fmt)
            ages.append(age.days / 365.25)
    age_deciles = np.percentile(ages, deciles)
    print 'Age deciles...'
    for di, ai in zip(deciles, age_deciles):
        print '    {:3}: {:.6}'.format(di, ai)
    print

    stats, group_stats = get_stats(population, args.transcript_file, args.delivery_file)

    # group = person_stats['group']
    # for field in ['viewed', 'trx', 'spend']:
    #     group_stats[group][field].append(person_stats[field])

    for field in ['received', 'viewed', 'trx', 'spend']:
        print field
        for group, gstats in group_stats.iteritems():
            data = gstats[field]
            print '{}: mean={:.6}, median={:.6}, stdev={:.6}'.format(group, np.mean(data), np.median(data), np.std(data))
        print

    return 0


def get_stats(population, transcript_file_name, delivery_file_name):
    treatments = dict()
    delimiter = '|'
    with open(delivery_file_name, 'r') as delivery_file:
        for line in delivery_file:
            data = line.strip().split(delimiter)
            if len(data) == 2:
                person_id, offer_id = data
                treatments[person_id] = offer_id

    stats = dict([(person_id, {'group': treatments.get(person_id, 'control'),
                               'received': 0,
                               'viewed': 0,
                               'trx': 0,
                               'spend': 0.00}) for person_id in population.people])

    fields = ['received', 'viewed', 'trx', 'spend']

    with open(transcript_file_name, 'r') as transcript_file:
        for line_number, line in enumerate(transcript_file):
            text = line.strip()
            if text != '':
                record = json.loads(text)
            else:
                continue

            if record['event'] == 'offer received':
                stats[record['person']]['received'] += 1

            if record['event'] == 'offer viewed':
                stats[record['person']]['viewed'] += 1
                if stats[record['person']]['viewed'] > 1:
                    print record['person']
                    raise

            if record['event'] == 'transaction':
                stats[record['person']]['trx'] += 1
                stats[record['person']]['spend'] += record['value']['amount']

    groups = set([person_stats['group'] for person_stats in stats.values()])
    group_names = dict()
    group_ctr = 0
    for group in groups:
        if group != 'control':
            group_name = 'offer_{}'.format(group_ctr)
            group_ctr += 1
        else:
            group_name = group
        group_names[group] = group_name
        print '{}: {}'.format(group_names[group], group)
    print

    group_stats = dict([(group_names[group], dict([(field, list()) for field in fields])) for group in groups])

    for person_stats in stats.values():
        group = person_stats['group']
        group_name = group_names[group]
        for field in fields:
            group_stats[group_name][field].append(person_stats[field])

    return stats, group_stats


def get_args():
    """Build arg parser and get command line arguments

    :return: parsed args namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--population-file", default="data/population.json", help="population file name")
    parser.add_argument("--transcript-file", default="data/transcript.json", help="transcript file name")
    parser.add_argument("--delivery-file",   default="data/delivery.csv",    help="delivery file name")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(get_args())
