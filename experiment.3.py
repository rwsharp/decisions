"""Run an experiment in which a portfolio of offers is deployed to a population."""

import logging
import unittest

import argparse
import os.path
import glob
import datetime
import numpy
import shutil

from externalities import World, Offer, Transaction, Event, Categorical
from person import Person
from population import Population

dt_fmt = '%Y%m%d'
now = datetime.datetime.strptime('20170718', dt_fmt)

def create_people_0(n):
    profile_optout_rate = 0.1
    min_tenure = 0
    max_tenure = 365
    mean_age = 365*30
    std_age = 365*2
    gender_rates = [0.49, 0.49, 0.02]
    min_income = 50000
    max_income = 75000
    beta = 1.0 / 0.0004
    g = lambda x: 1.0 / (1.0 + numpy.exp(-x))
    g_inv = lambda y: numpy.log(y / (1.0 - y))

    people = list()
    for i in range(n):
        became_member_on = (now - datetime.timedelta(days=numpy.random.choice(range(min_tenure, max_tenure)))).strftime(dt_fmt)

        if numpy.random.random() < 1.0 - profile_optout_rate:
            # must be at least 18 to join
            dob = (now - datetime.timedelta(days=int(max(365.25*18, (numpy.random.normal(mean_age, std_age)))))).strftime(dt_fmt)
            # three values + missing
            gender = numpy.random.choice(['M', 'F', 'O'], p=gender_rates)
            income = None
            for i in range(25):
                x = max(25000, numpy.random.exponential(beta))
                if min_income <= x <= max_income:
                    income = x
                    break
        else:
            dob = None
            gender = None
            income = None

        person_view_offer_sensitivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                                                    [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 0, 1, 1,
                                                     2])
        person_make_purchase_sensitivity = Categorical(
            ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
            [g_inv(1.0 / 24.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 1, 1])
        person_purchase_amount_sensitivity = Categorical(
            ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
             'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        people.append(Person(became_member_on,
                             dob=dob,
                             gender=gender,
                             income=income,
                             view_offer_sensitivity=person_view_offer_sensitivity,
                             make_purchase_sensitivity=person_make_purchase_sensitivity,
                             purchase_amount_sensitivity=person_purchase_amount_sensitivity))

    return people


def create_portfolio():
    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 0))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 1, 0))
    discount_a = Offer(0, valid_from=0, valid_until=7*24, difficulty=5, reward=2, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 1, 0))
    discount_b = Offer(0, valid_from=0, valid_until=7*24, difficulty=5, reward=2, channel=offer_channel, offer_type=offer_type)

    portfolio = (discount_a, discount_b)

    return portfolio


def mkdir_if_missing(path):
    if not os.path.isdir(path):
        if os.path.exists(path):
            raise ValueError('ERROR - the path name {} is already in use by a file or link.'.format(path))
        else:
            os.makedirs(path)


def assign_offers(population, deliveries_file_name, deliveries_log_file_name, control_fraction=0.25, delimiter='|', clean_path=True):

    if clean_path:
        data_file_names = glob.glob(os.path.join(population.deliveries_path, '*'))
        for data_file_name in data_file_names:
            os.remove(data_file_name)

    offer_ids = population.portfolio.keys()
    deliveries = list()
    for person_id in population.people.keys():
        # hold out a fraction of people as control
        if numpy.random.random() < 1.0 - control_fraction:
            # make random deliveries to the rest
            deliveries.append((person_id, numpy.random.choice(offer_ids)))

    # write the deliveries file
    with open(deliveries_file_name, 'w') as deliveries_file:
        for delivery in deliveries:
            print >> deliveries_file, delimiter.join(map(str, delivery))

    # make a copy of the delivery file
    shutil.copy(deliveries_file_name, deliveries_log_file_name)


def main(args):
    world = World(real_time_tick=0.000, world_time_tick=6)

    people = list()
    people.extend(create_people_0(1000))

    portfolio = create_portfolio()

    delivery_log_path = os.path.join(args.data_path, 'delivery_log')
    deliveries_path = os.path.join(args.data_path, 'delivery')
    transcripts_file_name = os.path.join(args.data_path, 'transcript.json')
    population_file_name = os.path.join(args.data_path, 'population.json')
    deliveries_file_name = os.path.join(deliveries_path, 'test_deliveries.csv')
    deliveries_log_file_name = os.path.join(delivery_log_path, 'test_deliveries.csv')

    # create directories if they don't already exist
    mkdir_if_missing(args.data_path)
    mkdir_if_missing(delivery_log_path)
    mkdir_if_missing(deliveries_path)

    # clean up from previous runs if there's data present
    for file_name in (transcripts_file_name, population_file_name, deliveries_file_name, deliveries_log_file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)

    population = Population(world,
                            people=people,
                            portfolio=portfolio,
                            deliveries_path=deliveries_path,
                            transcript_file_name=transcripts_file_name)

    with open(population_file_name, 'w') as population_file:
        print >> population_file, population.to_json()

    deliveries = assign_offers(population, deliveries_file_name, deliveries_log_file_name)

    population.simulate(n_ticks=4*21, n_proc=args.n_proc)

    return 0


def get_args():
    """Build arg parser and get command line arguments

    :return: parsed args namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="data",           help="data path file name")
    parser.add_argument("--n-proc",    default=1,      type=int, help="number of Processes to use for simulation")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(get_args())

