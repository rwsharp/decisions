"""Population

A population consists of a number of people (Person class entities). A population is initialized by randomly sampling
from a specified distribution of individual characteristics. The Population class also coordinates the reaction of
indiviudals to offers.
"""

# todo: popoulations change over time. Implement new member adds and churn

import logging
import unittest

import os.path
import glob
import shutil
import json
import multiprocessing
import time

from externalities import World, Offer, Transaction, Event, Categorical
from person import Person


class Population(object):
    """Create a collection of individuals and coordinate the reaction to offers."""

    def __init__(self,
                 world,
                 people=None,
                 portfolio=None,
                 deliveries_path='/deliveries',
                 transcript_file_name='/transcript.json'):
        """Initialize Population.

        Args:
        """

        assert isinstance(world, World), 'ERROR - world is not of World type.'
        self.world = world

        if people is None:
            self.people = dict()
        else:
            assert all(map(lambda p: isinstance(p, Person), people)), 'ERROR - some of the items in people are not Person type.'
            id_list = [p.id for p in people]
            assert len(set(id_list)) == len(id_list), 'ERROR - The same individual appears in people more than once.'

            self.people = dict([(p.id, p) for p in people])

        if portfolio is None:
            self.portfolio = dict()
        else:
            assert all(map(lambda o: isinstance(o, Offer), portfolio)), 'ERROR - some of the items in portfolio are not Offer type.'
            id_list = [o.id for o in portfolio]
            assert len(set(id_list)) == len(id_list), 'ERROR - The same offer appears in the portfolio more than once.'

            self.portfolio = dict([(o.id, o) for o in portfolio])

        self.deliveries_path = deliveries_path
        self.transcript_file_name = transcript_file_name

        logging.info('Population initialized')


    def to_serializable(self):
        """Create a serializable representation."""
        population_dict = {'world':           self.world.to_serializable(),
                           'people':          [person.to_serializable() for person in self.people.values()],
                           'portfolio':       [offer.to_serializable() for offer in self.portfolio.values()],
                           'deliveries_path': self.deliveries_path,
                           'transcript_file_name':     self.transcript_file_name}

        return population_dict


    @staticmethod
    def from_dict(population_dict):

        population = Population(World.from_dict(population_dict.get('world')),
                                people=[Person.from_dict(person_dict) for person_dict in population_dict.get('people')],
                                portfolio=[Offer.from_dict(offer_dict) for offer_dict in population_dict.get('portfolio')],
                                deliveries_path=population_dict.get('deliveries_path'),
                                transcript_file_name=population_dict.get('transcript_file_name'))

        return population


    @staticmethod
    def from_json(json_string):
        """Read a population from json format."""
        # todo: update other from_json methods to do file check as well
        if os.path.isfile(json_string):
            # it's json in a file
            with open(json_string, 'r') as population_file:
                population_dict = json.load(population_file)
        else:
            # it's a json string
            population_dict = json.loads(json_string)

        population = Population.from_dict(population_dict)

        return population


    def to_json(self):
        """Create a json representation."""
        json_string = json.dumps(self.to_serializable())

        return json_string


    def simulate(self, n_ticks, n_proc=1):
        """Coordinate the simulation of individual reactions to offers.

        Simulation proceeds for n steps. The simulated duration of each step is a property of the World.
        """
        start = time.time()
        last = start
        for t in range(n_ticks):
            if t % int(n_ticks/10.0) == 0:
                now = time.time()
                print '{} of {} days finished (time = {:.4}, delta = {:.4}).'.format(self.world.world_time/24.0, n_ticks*self.world.world_time_tick/24.0, round(now-start,3), round(now-last,3))
                last = now

            deliveries = self.read_deliveries(cleanup=True)
            if n_proc > 1:
                self.update_people(deliveries, n_proc)
            else:
                self.update_people_serial(deliveries)
            self.world.update()
            self.report()


    def report(self):
        logging.info('Report')


    def read_deliveries(self, cleanup=False, delimiter='|'):
        """Read offer data if present in offers_path.

        An offers file should have one two columns separated by a delimiter (a .csv)
        There should be no header
        schema: 'recipient_id|offer_id'
        """

        deliveries = dict()
        delivery_file_names = glob.glob(os.path.join(self.deliveries_path, '*'))
        for delivery_file_name in delivery_file_names:
            if os.path.isfile(delivery_file_name):
                with open(delivery_file_name, 'r') as delivery_file:
                    for line in delivery_file:
                        data = line.strip().split(delimiter)
                        if len(data) == 2:
                            recipient_id, offer_id = data
                            deliveries[recipient_id] = offer_id

            if cleanup:
                os.remove(delivery_file_name)

        return deliveries


    def update_people(self, deliveries, n_proc=2):
        """Simulate a single timestep for everybody in the population."""
        self.deliver_offers(deliveries)

        n_people = len(self.people)
        chunk_size = n_people/n_proc

        people_chunks = [self.people.keys()[i:min(i + chunk_size, n_people)] for i in xrange(0, n_people, chunk_size)]
        all_people = set(reduce(lambda a, b: a+b, people_chunks))
        assert all_people == set(self.people.keys()), 'ERROR - Some people were lost during chunking.'
        procs = [multiprocessing.Process(target=self.people_loop, args=(people_chunks[p],)) for p in range(n_proc)]
        for proc in procs:
            proc.start()

        for proc in procs:
           proc.join()


    def people_loop(self, person_id_list):
        """Do the person update for everybody in the id list - handy form for parallelism."""
        transcript = list()
        for person_id in person_id_list:
            person = self.people[person_id]
            transcript.extend(person.update(self.world))
        self.write_to_transcript_file(transcript)


    def update_people_serial(self, deliveries):
        """Simulate a single timestep for everybody in the population."""

        self.deliver_offers(deliveries)

        transcript = list()
        for id, person in self.people.iteritems():
            transcript.extend(person.update(self.world))
        self.write_to_transcript_file(transcript)


    def write_to_transcript_file(self, transcript):
        with open(self.transcript_file_name, 'a') as transcript_file:
            print >> transcript_file, '\n'.join(transcript)

    def deliver_offers(self, deliveries):
        """Go through the offer list and deliver to recipients."""

        transcript = list()

        for recipient_id, offer_id in deliveries.iteritems():
            recipient = self.people.get(recipient_id)
            if recipient is None:
                message = 'WARNING - recipient {} is not in the population, cannot deliver.'.format(recipient_id)
                logging.warning(message)
            else:
                offer = self.portfolio.get(offer_id)
                if offer is None:
                    message = 'WARNING - offer {} is not in the portfolio, cannot deliver.'.format(offer_id)
                    logging.warning(message)
                else:
                    transcript.extend(recipient.receive_offer(self.world, offer))

        self.write_to_transcript_file(transcript)


    def read_offer_portfolio(self, portfolio_file_name):
        """Read in an offer portfolio from a file.

        An offer portfolio file contains one json object per line.
        Each json object represents a single offer.
        """

        with open(portfolio_file_name, 'r') as portfolio_file:
            for line in portfolio_file:
                offer_json = line.strip()

                # skip blank lines
                if offer_json != '':
                    offer = Offer.from_json(offer_json)
                    offer_id = offer['id']
                    if offer_id not in self.offer_portfolio:
                        self.offer_portfolio[id] = offer
                    else:
                        raise ValueError('ERROR - Offer id {} is not unique. It is already present in the offer portfolio.'.format(offer_id))


    def read_population(self, population_file_name):
        """Read in a population from a file.

        A population file contains one json object per line.
        Each json object represents a Person.
        """

        with open(population_file_name, 'r') as population_file:
            for line in population_file:
                person_json = line.strip()

                # skip blank lines
                if person_json != '':
                    person = Person.from_json(person_json)
                    person_id = person['id']
                    if person_id not in self.people:
                        self.people[id] = person
                    else:
                        raise ValueError('ERROR - Person id {} is not unique. It is already present in the population.'.format(person_id))


class TestPopulation(unittest.TestCase):
    """Test class for Population."""

    def setUp(self):
        self.world = World(real_time_tick=0.200)

        person_0 = Person('20170101')
        person_1 = Person('20170202')
        person_2 = Person('20170707')

        offer_a = Offer(0)
        offer_b = Offer(1)

        self.delimiter = '|'
        deliveries_path = 'data/delivery'
        self.deliveries_file_name = 'data/test_deliveries.csv'
        self.deliveries = [(person_0.id, offer_a.id),
                           (person_1.id, offer_a.id),
                           (person_2.id, offer_b.id)]

        with open(self.deliveries_file_name, 'w') as deliveries_file:
            for delivery in self.deliveries:
                print >> deliveries_file, self.delimiter.join(map(str, delivery))

        self.population = Population(self.world,
                                     people=(person_0, person_1, person_2),
                                     portfolio=(offer_a, offer_b),
                                     deliveries_path=deliveries_path,
                                     transcript_file_name='data/transcript.json')


    def tearDown(self):
        if os.path.isfile(self.deliveries_file_name):
            os.remove(self.deliveries_file_name)


    def test_simulate(self):
        # place delivery file in deliveries folder
        data_file_names = glob.glob(os.path.join(self.population.deliveries_path, '*'))
        for data_file_name in data_file_names:
            os.remove(data_file_name)
        shutil.copy(self.deliveries_file_name, self.population.deliveries_path)

        # run simulation
        self.population.simulate(n_ticks=5)
        self.assertTrue(True)


    def test_read_deliveries(self):
        data_file_names = glob.glob(os.path.join(self.population.deliveries_path, '*'))
        for data_file_name in data_file_names:
            os.remove(data_file_name)
        shutil.copy(self.deliveries_file_name, self.population.deliveries_path)
        deliveries = self.population.read_deliveries(cleanup=True)
        self.assertTrue(len(deliveries) > 0)
        data_file_names = glob.glob(os.path.join(self.population.deliveries_path, '*'))
        self.assertTrue(len(data_file_names) == 0)
        
    
    def test_serializaton(self):
        population_dict = self.population.to_serializable()
        population_reconstituted = Population.from_dict(population_dict)
        population_reconstituted_dict = population_reconstituted.to_serializable()
        self.assertTrue(population_reconstituted_dict == population_dict)

        population_json = self.population.to_json()
        population_reconstituted = Population.from_json(population_json)
        population_reconstituted_dict = population_reconstituted.to_serializable()
        self.assertTrue(population_reconstituted_dict == population_dict)

