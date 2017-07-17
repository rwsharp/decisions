"""Population

A population consists of a number of people (Person class entities). A population is initialized by randomly sampling
from a specified distribution of individual characteristics. The Population class also coordinates the reaction of
indiviudals to offers.
"""

# todo: popoulations change over time. Implement new member adds and churn

import logging
import unittest

from externalities import World
from person import Person


class Population(object):
    """Create a collection of individuals and coordinate the reaction to offers."""

    def __init__(self,
                 world,
                 people,
                 offers_path='/offers',
                 events_path='/events'):
        """Initialize Population.

        Args:
        """

        assert isinstance(world, World), 'ERROR - world is not of World type.'

        assert all(map(lambda p: isinstance(p, Person), people)), 'ERROR - some of the items in people are not Person type.'
        id_list = [p.id for p in people]
        assert len(set(id_list)) == len(id_list), 'ERROR - The same individual appears in people more than once.'

        self.world = world
        self.population = dict([(p.id, p) for p in people])
        self.offers_path = offers_path
        self.events_path = events_path
        self.offer_portfolio = set()

        logging.info('Population initialized')


    def simulate(self, n_ticks):
        """Coordinate the simulation of individual reactions to offers.

        Simulation proceeds for n steps. The simulated duration of each step is a property of the World.
        """
        for t in range(n_ticks):
            offers = self.get_offers(cleanup=True)
            self.update_people(offers)
            self.world.update()
            self.report()


    def read_offer_portfolio(self, portfolio_file_name):
        """Read in an offer portfolio from a file.

        An offer portfolio file contains one json object per line.
        Each json object represents a single offer.
        """




    def get_offer_delivery_instructions(self, cleanup=False):
        """Read offer data if present in offers_path.

        An offers file should have one json object per line. The json object has the following format...
        {'recipient_id': uuid_of_Person, 'offer_id': uuid_of_Offer}
        """


    def update_people(self, offers):
        """Simulate a single timestep for everybody in the population."""

        self.deliver_offers(offers)

        # todo: parallelize this loop
        for id, person in self.people.iteritems():
            person.update(self.world)


    def deliver_offers(self, offers):
        """Go through the offer list and deliver to recipients."""

        for recipient_id, offer in offers:
            recipient = self.people.get(recipient_id)
            if recipient is None:
                message = 'WARNING - recipient {} is not in the population, cannot deliver.'.format(recipient_id)
                logging.warning(message)
            else:
                recipient.receive_offer(self.world, offer)


class TestPopulation(unittest.TestCase):
    """Test class for Population."""

    def setUp(self):
        self.Population = Population()