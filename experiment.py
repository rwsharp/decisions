"""Run an experiment in which a portfolio of offers is deployed to a population."""

import logging
import unittest

import os.path
import glob
import shutil
import json
import datetime
import numpy

from externalities import World, Offer, Transaction, Event, Categorical
from person import Person
from population import Population

dt_fmt = '%Y%m%d'
now = datetime.datetime.strptime('20170718', dt_fmt)

def create_people(n):

    people = list()
    for i in range(n):
        dob = (now - datetime.timedelta(days=numpy.random.poisson(365*25))).strftime(dt_fmt)
        people.append(Person(dob))

    return people


def create_portfolio():
    offer_a = Offer(0)
    offer_b = Offer(1)

    portfolio = (offer_a, offer_b)

    return portfolio


def assign_offers(population, control_fraction=0.5, delimiter='|', clean_path=True):

    if clean_path:
        data_file_names = glob.glob(os.path.join(population.deliveries_path, '*'))
        for data_file_name in data_file_names:
            os.remove(data_file_name)

    deliveries_file_name = os.path.join(population.deliveries_path, 'test_deliveries.csv')

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


def main():
    world = World(real_time_tick=0.000)
    people = create_people(1000)
    portfolio = create_portfolio()

    deliveries_path = 'data/delivery'
    transcripts_file_name = 'data/transcript.json'

    population = Population(world,
                            people=people,
                            portfolio=portfolio,
                            deliveries_path=deliveries_path,
                            transcript_file_name=transcripts_file_name)

    deliveries = assign_offers(population)

    population.simulate(n_ticks=5)

    return 0


if __name__ == '__main__':
    main()

