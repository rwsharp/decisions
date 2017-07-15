"""World - for storing and updating the state of the world.

A World consists of a population that may react to offers. Time advances discreetly on two scales: the
real interval between "ticks" and the amount of simulation time in that interval. For example, one second in real time
might correspond to 24 hours in simulation time. Once started, the World continues to run until the simulation receives
a "stop" instruction (or is shut down by the system). During each tick of the simulation, the world reads in new
offer instructions, simulates the response of the population, and writes the resulting events (individual actions) to
files. A person or program wishing to interact with the World does so by writing offer files or reading events files
from the specified file locations.
"""

import logging
import unittest

from population import Population

class World(object):
    """Coordinate the delivery of offers to a population and simulation of the reaction."""
    def __init__(self,
                 realtime_tick=1.0,
                 simtime_tick=24.0,
                 offers_path='/offers',
                 events_path='/events'):
        """Initialize World.

        Args:
            realtime_tick: wallclock time in seconds between ticks of the simulation
            simtime_tick:  amount of time in hours that passes in the simulated World between ticks of the simulation
            offers_path:   path to folder containing offer instructions
            events_path:   path to folder where simulated events are stored
        """
        # World parameters
        self.realtime_tick = realtime_tick
        self.simtime_tick = simtime_tick
        self.offers_path = offers_path
        self.events_path = events_path

        # Population parameters
        self.population_config = population_config
        self.population = Population(population_config)

        logging.info('World initialized')

    def wait(self):
        """Sit idle during the realtime tick period."""
        raise NotImplementedError()


    def read_offers(self):
        """Read offer data if present in offers_path."""
        raise NotImplementedError()


    def clean_offers(self):
        """Clean the offers_path after reading."""
        raise NotImplementedError()


    def simulate(self):
        """Simulate the actions of the population during the tick."""
        raise NotImplementedError()


    def report(self):
        """Write simulation data to files in the events_path."""
        raise NotImplementedError()


    def run(self, n_ticks=None):
        """Run the world for n_ticks."""
        for t in range(n_ticks):
            self.wait()
            self.read_offers()
            self.clean_offers()
            self.simulate()
            self.report()

        raise NotImplementedError()





class TestWorld(unittest.TestCase):
    """Test class for World."""

    def setUp(self):
        self.World = World()