"""Externalities - Constants, classes, and methods for storing and updating the state of the world and events that exert
external influences on a Person.

Methods:
Classes: Constants, World,
"""

import logging
import unittest

import numpy

##################
# GLOBAL CONSTANTS
##################

class Constants(object):
    """Universal constants"""

    BEGINNING_OF_TIME = 0
    END_OF_TIME = 999999999


class World(object):
    """Externalities - Constants, classes, and methods for storing and updating the state of the world.

    A World consists of a population that may react to offers. Time advances discreetly on two scales: the
    real interval between "ticks" and the amount of simulation time in that interval. For example, one second in real time
    might correspond to 24 hours in simulation time. Once started, the World continues to run until the simulation receives
    a "stop" instruction (or is shut down by the system). During each tick of the simulation, the world reads in new
    offer instructions, simulates the response of the population, and writes the resulting events (individual actions) to
    files. A person or program wishing to interact with the World does so by writing offer files or reading events files
    from the specified file locations.
    """
    """Coordinate the delivery of offers to a population and simulation of the reaction."""

    def __init__(self,
                 world_time = Constants.BEGINNING_OF_TIME,
                 real_time_tick=1.0,
                 world_time_tick=24,
                 offers_path='/offers',
                 events_path='/events'):
        """Initialize World.

        Args:
            real_time_tick: wallclock time in seconds between ticks of the simulation
            world_time_tick:  amount of time in hours that passes in the simulated World between ticks of the simulation
            offers_path:   path to folder containing offer instructions
            events_path:   path to folder where simulated events are stored
        """
        # World parameters
        self.real_time_tick = real_time_tick
        self.world_time_tick = world_time_tick
        self.offers_path = offers_path
        self.events_path = events_path

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
        """Simulate the world external to the individuals of a population during the tick.

        At a minimum this increases simulation time by one tick."""

        self.world_time += self.world_time_tick

        if self.world_time > Constants.END_OF_TIME:
            raise ValueError('ERROR - You are literally out of time.')


    def report(self):
        """Write simulation data to files in the events_path."""
        raise NotImplementedError()

        raise NotImplementedError()


class TestWorld(unittest.TestCase):
    """Test class for World."""

    def setUp(self):
        self.World = World()
        

class Categorical(object):
    """A categorical variable class.

    For example, a Person can have weighted membership in each category of a segment variable. The names and weights are
    represented by separate numpy arrays. These variables are often used in a one-hot encoding context, so by using
    separate name and weight arrays instead of a single dictionary, we can do linear algebra with teh weights without
    the overhead of pulling values out of a dictionary in a specified order.
    """
    names = numpy.array(tuple())
    weights = numpy.array(tuple())

    def __init__(self, names=None, weights=None):
        """Initialize Segment."""

        if names is not None:
            assert len(set(names)) == len(names), 'ERROR - Not all names are unique.'
            self.names = numpy.array(names)

            if weights is None:
                self.weights = numpy.ones(len(names), dtype=numpy.int)
            else:
                assert len(names) == len(weights), 'ERROR - The number of names does not match the number of weights.'
                self.weights = numpy.array(weights)


    def get(self, name, default=None):
        """Indexing by segment name."""
        locations = numpy.where(self.names == name)[0]
        n = len(locations)
        if n == 1:
            # found the match, return value
            # because names are unique, there can only be one index if any
            return self.weights[locations[0]]
        elif n == 0:
            # no match, return default value
            return default
        else:
            # because names are unique, there can only be one index if any
            # so something went wrong if we get here
            raise ValueError('ERROR - Names are unique, so only only one match is possible, but received more: {}'.format(locations[0]))


    def get_index(self, name, default=None):
        """Get the index of the given name."""
        locations = numpy.where(self.names == name)[0]
        n = len(locations)
        if n == 1:
            # found the match, return index
            # because names are unique, there can only be one index if any
            return locations[0]
        elif n == 0:
            # no match, return default value
            return default
        else:
            # because names are unique, there can only be one index if any
            # so something went wrong if we get here
            raise ValueError('ERROR - Names are unique, so only only one match is possible, but received more: {}'.format(locations[0]))


    def set(self, name, value):
        """Set value of named component."""
        locations = numpy.where(self.names == name)[0]
        n = len(locations)
        if n == 1:
            # found the match, set value
            self.weights[locations[0]] = value
        elif n > 1:
            # because names are unique, there can only be one index if any
            # so something went wrong if we get here
            raise ValueError('ERROR - Names are unique, so only only one match is possible, but received more: {}'.format(locations[0]))


    def set_equal(self, other):
        """A limited equality operator to assign the names and values of other to self, resizing self if necessary."""

        self.names = other.names
        self.weights = other.weights


    def compare_equality(self, other):
        """Check if two Categoricals have the same names and weights in any order."""
        if set(self.names) == set(other.names):
            self_equals_other = all([self.get(name) == other.get(name) for name in self.names])
        else:
            self_equals_other = False

        return self_equals_other


    def compare_strict_equality(self, other):
        """Check if two Categoricals have the same names and weights in the same order."""
        if all(self.names == other.names):
            self_equals_other = all(self.weights == other.weights)
        else:
            self_equals_other = False

        return self_equals_other


    def set_order(self, names):
        """Reorder the names and weights of a categorical to match the given order in the names argument."""
        assert set(self.names) == set(names), 'ERROR - The given set of names {} does not match the set of existing names {}'.format(names, self.names)
        assert len(set(names)) == len(names), 'ERROR - The given set of names {} has duplicate values'.format(names)

        self.weights = numpy.array([self.get(name) for name in names])
        self.names = numpy.array(names)


class TestCategorical(unittest.TestCase):
    """Test class for Categorical."""

    def setUp(self):
        self.cat_0 = Categorical(names=('a', 'b', 'c', 'd'), weights=(1, 1, 3, 1))
        self.cat_1 = Categorical(names=('d', 'c', 'b', 'a'), weights=(4, 1, 2, 1))
        self.cat_2 = Categorical(names=('w', 'x', 'y', 'z'), weights=(1, 1, 3, 1))
        self.cat_3 = Categorical(names=('a', 'b', 'c', 'd'), weights=(1, 2, 3, 4))
        self.cat_4 = Categorical(names=('z', 'y', 'x', 'w'), weights=(1, 3, 1, 1))

        # cat_0 not equal cat_1,2,3,4
        # cat_2 soft equal cat_4
        # cat_0 * cat_1 strict equal cat 3
        # sum(cat_0 * cat_1) = 10


    def test_equality_comparisons(self):
        comparisons = (not self.cat_0.compare_equality(self.cat_1),
                       not self.cat_0.compare_equality(self.cat_2),
                       not self.cat_0.compare_equality(self.cat_3),
                       not self.cat_0.compare_equality(self.cat_4),
                       self.cat_2.compare_equality(self.cat_4),
                       self.cat_0.compare_strict_equality(self.cat_0)
                       )

        self.assertTrue(all(comparisons))


    def test_set(self):
        x = self.cat_0
        y = self.cat_1

        for name in x.names:
            y.set(name, x.get(name))

        self.assertTrue(x.compare_equality(y))


    def test_set_equal(self):
        """It's critical that categoricals with the same names in different order are correctly aligned."""

        c = Categorical()
        c.set_equal(self.cat_0)

        self.assertTrue(self.cat_0.compare_strict_equality(c))


    def test_mixed_dot_product(self):
        """A mixed dot product is the regular dot product of two numpy arrays where one of the arrays is partially
        defined by the ordered weights of a Categorical.
        """

        x = self.cat_0
        y = self.cat_1
        y.set_order(x.names)
        z = self.cat_3
        z.set_order(x.names)

        a = numpy.concatenate((numpy.array((0,)), x.weights))
        b = numpy.concatenate((numpy.array((0,)), y.weights))
        c = numpy.concatenate((numpy.array((0,)), z.weights))

        new_names = numpy.concatenate((numpy.array(('null',)), x.names))

        prod_ab = a * b

        r = Categorical(names=new_names, weights=prod_ab)
        s = Categorical(names=new_names, weights=c)

        self.assertTrue(r.compare_strict_equality(s))
        self.assertTrue(numpy.sum(r.weights) == 10)


class Event(object):
    """Events are the actions and interactions that occur with respect to a Person.

    A Person's history is a timeseries of Events combined with a timeseries of attribute changes. We treat them
    separately, instead of creating a "state_change" event, because state_changes are intrinsic to the individual,
    whereas events are extrinsic.

    example of an offer event
        {'type': 'offer',
         'timestamp': 12345,
         'value': {'valid_from': 1235, 'valid_to': 23456,
                   'difficulty': 2, 'reward': 4,
                   'channel_web': 1 / 0, 'channel_mobile'; 1 / 0,
                   'type_bogo': 1 / 0, 'type_discount': 1 / 0, ...}}
    """

    timestamp = None
    value = None

    def __init__(self, timestamp):
        """Initialize Person.

        Args:
        """

        self.timestamp = timestamp


class TestEvent(unittest.TestCase):
    """Test class for Event."""

    def setUp(self):
        timestamp = 12345
        self.event = Event(timestamp)


    def test_init(self):
        print self.event.__dict__
        self.assertTrue(self.event)


class Offer(Event):
    """Offer events occur when a Person receives an offer and offer an incentive to make a purchase.

    timestamp: the time the offer is received
    valid_from: the start of the offer validity period
    valid_until: the end of the offer validity period
    difficulty: the effort needed to win
    reward: the reward for winning
    channel_web: flag for web offer
    channel_mobile: flag for mobile offer (an offer can be multichannel, but it can't be zero channel)
    type_bogo: offer type is buy-one-get-one (difficulty = reward)
    type_discount: offer type is a discount (difficulty = normal price, reward = amount returned, typically a fraction of the difficulty)
    type_informational: an informative offer, no call to action, no reward (difficulty = reward = 0)

    example of an offer event
        {'timestamp': 12345,
         'valid_from': 1235,
         'valid_to': 23456,
         'difficulty': 2,
         'reward': 4,
         'channel_web': 1 / 0,
         'channel_mobile'; 1 / 0,
         'type_bogo': 1 / 0,
         'type_discount': 1 / 0, ...}
    """

    valid_from = None
    valid_until = None
    difficulty = None
    reward = None

    channel = Categorical(('web', 'email', 'mobile'))
    type = Categorical(('bogo', 'discount', 'informational'))

    def __init__(self, timestamp_received, **kwargs):
        self.timestamp = timestamp_received
        self.valid_from = kwargs.get('valid_from', Constants.BEGINNING_OF_TIME)
        self.valid_until = kwargs.get('valid_until', Constants.END_OF_TIME)
        self.difficulty = kwargs.get('difficulty', 0)
        self.reward = kwargs.get('reward', 0)

        x = kwargs.get('channel')
        if x is not None:
            self.channel.set_equal(x)

        x = kwargs.get('type')
        if x is not None:
            self.type.set_equal(x)

        assert numpy.sum(self.channel.weights) > 0, 'ERROR - offer must have at least one channel'
        assert numpy.sum(self.type.weights) == 1,   'ERROR - offer must have exactly one type'


    def is_active(self, current_time):
        """Determine if the offer is valid at the current time."""
        return True if self.valid_from <= current_time <= self.valid_until else False


class TestOffer(unittest.TestCase):
    """Test class for Offer."""

    def setUp(self):
        timestamp = 12345

        offer_channel = Categorical(('web', 'email', 'mobile'), (0, 1, 1))
        offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 0, 1))

        self.offer = Offer(timestamp, channel=offer_channel, type=offer_type)

    def test_init(self):
        print self.offer.__dict__
        self.assertTrue(self.offer)


class Transaction(Event):
    """Offer events occur when a Person receives an offer and offer an incentive to make a purchase.

    timestamp: the time the offer is received
    amount: the amount of the purchase
    """

    def __init__(self, timestamp_received, **kwargs):
        self.timestamp = timestamp_received
        self.amount = kwargs.get('amount', None)


class TestTransaction(unittest.TestCase):
    """Test class for Transaction."""

    def setUp(self):
        timestamp = 12345
        amount = 1.0

        self.transaction = Transaction(timestamp, amount=amount)

