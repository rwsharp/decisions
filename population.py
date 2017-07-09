"""Population

A population consists of a number of people (Person class entities). A population is initialized by randomly sampling
from a specified distribution of individual characteristics. The Population class also coordinates the reaction of
indiviudals to offers.
"""

# todo: popoulations change over time. Implement new member adds and churn

import logging
import unittest


class Population():
    """Create a collection of individuals and coordinate the reaction to offers."""
    def __init__(self):
        """Initialize Population.

        Args:
        """
        raise NotImplementedError()

        self.people = set()

        logging.info('Population initialized')


    def simulate(self):
        """Coordinate the simulation of individual reactions to offers."""
        raise NotImplementedError()




class TestPopulation(unittest.TestCase):
    """Test class for Population."""

    def setUp(self):
        self.Population = Population()