"""Externalities - Constants, classes, and methods for storing and updating the state of the world and events that exert
external influences on a Person.

Methods:
Classes: Constants, World,
"""

import logging
import unittest

import types
import time
import uuid
import json

import numpy


numeric_types = (types.IntType, types.LongType, types.FloatType)

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
                 world_time=Constants.BEGINNING_OF_TIME,
                 real_time_tick=1.000,
                 world_time_tick=1):
        """Initialize World.

        Args:
            real_time_tick: wallclock time in seconds between ticks of the simulation
            world_time_tick:  amount of time in hours that passes in the simulated World between ticks of the simulation
            offers_path:   path to folder containing offer instructions
            events_path:   path to folder where simulated events are stored
        """
        # World parameters
        self.world_time = world_time
        self.real_time_tick = real_time_tick
        self.world_time_tick = world_time_tick

        logging.info('World initialized')


    def to_serializable(self):
        """Create a serializable representation."""
        world_dict = {'world_time':      self.world_time,
                      'real_time_tick':  self.real_time_tick,
                      'world_time_tick': self.world_time_tick}

        return world_dict


    @staticmethod
    def from_dict(world_dict):

        world = World(     world_time=world_dict.get('world_time', Constants.BEGINNING_OF_TIME),
                           real_time_tick=world_dict.get('real_time_tick', 1.0),
                           world_time_tick=world_dict.get('world_time_tick', 1))

        return world


    @staticmethod
    def from_json(json_string):
        world_dict = json.loads(json_string)
        world = World.from_dict(world_dict)

        return world


    def to_json(self):
        """Create a json representation."""
        json_string = json.dumps(self.to_serializable())

        return json_string


    def clean_offers(self):
        """Clean the offers_path after reading."""
        raise NotImplementedError()


    def update(self):
        """Simulate the world external to the individuals of a population during the tick.

        At a minimum this increases simulation time by one tick."""

        time.sleep(self.real_time_tick)

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
        self.world = World()


    def test_serializaton(self):
        world_dict = self.world.to_serializable()
        world_reconstituted = World.from_dict(world_dict)
        world_reconstituted_dict = world_reconstituted.to_serializable()
        self.assertTrue(world_reconstituted_dict == world_dict)

        world_json = self.world.to_json()
        world_reconstituted = World.from_json(world_json)
        world_reconstituted_dict = world_reconstituted.to_serializable()
        self.assertTrue(world_reconstituted_dict == world_dict)



class Categorical(object):
    """A categorical variable class.

    For example, a Person can have weighted membership in each category of a segment variable. The names and weights are
    represented by separate numpy arrays. These variables are often used in a one-hot encoding context, so by using
    separate name and weight arrays instead of a single dictionary, we can do linear algebra with teh weights without
    the overhead of pulling values out of a dictionary in a specified order.
    """
    names = numpy.array(tuple())
    weights = numpy.array(tuple())
    zeros = numpy.zeros(tuple())
    ones = numpy.ones(tuple())

    def __init__(self, names=None, weights=None):
        """Initialize Segment."""

        if names is not None:
            assert len(set(names)) == len(names), 'ERROR - Not all names are unique.'
            self.names = numpy.array(names)

            # these are handy defaults to have around - but consider dropping if there are space issues
            self.zeros = numpy.zeros(len(names), dtype=numpy.int)
            self.ones = numpy.ones(len(names), dtype=numpy.int)

            if weights is None:
                self.weights = self.ones.copy()
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

        self.names = other.names.copy()
        self.weights = other.weights.copy()
        self.zeros = other.zeros.copy()
        self.ones = other.ones.copy()


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


    def compare_names(self, other):
        """True if both Categoricals have the same names in any order, else False."""
        if set(self.names) == set(other.names):
            self_same_names_as_other = True
        else:
            self_same_names_as_other = False

        return self_same_names_as_other


    def set_order(self, names):
        """Reorder the names and weights of a categorical to match the given order in the names argument."""
        assert set(self.names) == set(names), 'ERROR - The given set of names {} does not match the set of existing names {}'.format(names, self.names)
        assert len(set(names)) == len(names), 'ERROR - The given set of names {} has duplicate values'.format(names)

        self.weights = numpy.array([self.get(name) for name in names])
        self.names = numpy.array(names)


    def to_serializable(self):
        categorical_dict = {'names':   list(self.names),
                            'weights': list(self.weights)}

        return categorical_dict

    @staticmethod
    def from_dict(categorical_dict):
        cat = Categorical(names  =categorical_dict.get('names'), \
                          weights=categorical_dict.get('weights'))

        return cat


    def to_json(self):
        """Create a json representation of the Categorical."""
        json_string = json.dumps(self.to_serializable())

        return json_string


    @staticmethod
    def from_json(json_string):
        """Turn json into a Categorical."""
        categorical_dict = json.loads(json_string)
        cat = Categorical.from_dict(categorical_dict)

        return cat


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


    def test_serializaton(self):
        cat_dict = self.cat_0.to_serializable()
        cat_reconstituted = Categorical.from_dict(cat_dict)
        cat_reconstituted_dict = cat_reconstituted.to_serializable()
        self.assertTrue(cat_reconstituted_dict == cat_dict)

        cat_json = self.cat_0.to_json()
        cat_reconstituted = Categorical.from_json(cat_json)
        cat_reconstituted_dict = cat_reconstituted.to_serializable()
        self.assertTrue(cat_reconstituted_dict == cat_dict)


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

    def __init__(self, timestamp, **kwargs):
        """Initialize Person.

        Args:
        """

        valid_kwargs = set(('id', 'value'))
        kwargs_name_set = set(kwargs.keys())
        assert kwargs_name_set.issubset(valid_kwargs), 'ERROR - Invalid kwargs: {}'.format(kwargs_name_set.difference(valid_kwargs))

        self.type = 'event'
        self.id = kwargs.get('id') if kwargs.get('id') is not None else uuid.uuid4().hex
        self.timestamp = timestamp
        self.value = kwargs.get('value', None)


    #################################################################################
    # Event Specific Methods - Each Event subclass should implement its own versions.
    #################################################################################

    def to_serializable(self):
        """Create a serializable representation."""
        event_dict = {'type':      self.type,
                      'id':        self.id,
                      'timestamp': self.timestamp,
                      'value':     self.value}

        return event_dict


    @staticmethod
    def from_dict(event_dict):
        assert event_dict.get('type') == 'event', 'ERROR - Dictionary must assert that it represents an Event, but type is {}.'.format(event_dict.get('type'))
        event = Event(            event_dict.get('timestamp'),
                                  id   =event_dict.get('id'),
                                  value=event_dict.get('value'))

        return event


    @staticmethod
    def from_json(json_string):
        event_dict = json.loads(json_string)
        event = Event.from_dict(event_dict)

        return event


    #######################################################
    # Generic Event Methods - Used by all Event subclasses.
    #######################################################

    def to_json(self):
        """Create a json representation."""
        json_string = json.dumps(self.to_serializable())

        return json_string




class TestEvent(unittest.TestCase):
    """Test class for Event."""

    def setUp(self):
        timestamp = 12345
        value = 0.00
        self.event = Event(timestamp, value=value)


    def test_init(self):
        print self.event.__dict__
        self.assertTrue(self.event)


    def test_serializaton(self):
        event_dict = self.event.to_serializable()
        event_reconstituted = Event.from_dict(event_dict)
        event_reconstituted_dict = event_reconstituted.to_serializable()
        self.assertTrue(event_reconstituted_dict == event_dict)

        event_json = self.event.to_json()
        event_reconstituted = Event.from_json(event_json)
        event_reconstituted_dict = event_reconstituted.to_serializable()
        self.assertTrue(event_reconstituted_dict == event_dict)


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


    def __init__(self, timestamp_received, **kwargs):
        default_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
        default_offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 0, 1))

        valid_kwargs = {'id', 'valid_from', 'valid_until', 'difficulty', 'reward', 'progress', 'completed', 'channel', 'offer_type'}
        kwargs_name_set = set(kwargs.keys())
        assert kwargs_name_set.issubset(valid_kwargs), 'ERROR - Invalid kwargs: \n{}\n{}\n{}'.format(kwargs_name_set, valid_kwargs, kwargs_name_set - valid_kwargs)

        self.type = 'offer'
        self.id = kwargs.get('id') if kwargs.get('id') is not None else uuid.uuid4().hex
        self.timestamp = timestamp_received
        self.valid_from = kwargs.get('valid_from', Constants.BEGINNING_OF_TIME)
        self.valid_until = kwargs.get('valid_until', Constants.END_OF_TIME)

        # A customer must spend an amount greater than 'difficulty' during the validity period to earn 'reward'
        # 'progress' is the amount spend during the validity period (and should be updated iteratively)
        # 'completed' is a convenience flag indicating whether the offer was completed
        self.difficulty = kwargs.get('difficulty', 0.00)
        assert isinstance(self.difficulty, numeric_types), 'ERROR - difficulty must be numeric'
        self.reward = kwargs.get('reward', 0.00)
        assert isinstance(self.reward, numeric_types), 'ERROR - reward must be numeric'
        self.progress = kwargs.get('progress', 0.00)
        assert isinstance(self.progress, numeric_types), 'ERROR - progress must be numeric'
        self.completed = kwargs.get('completed', False)
        assert isinstance(self.completed, types.BooleanType), 'ERROR - completed must be Boolean'

        self.channel = default_channel
        x = kwargs.get('channel')
        if x is not None:
            self.channel.set_equal(x)

        self.offer_type = default_offer_type
        x = kwargs.get('offer_type')
        if x is not None:
            self.offer_type.set_equal(x)

        assert numpy.sum(self.channel.weights) > 0, 'ERROR - offer must have at least one channel'
        assert numpy.sum(self.offer_type.weights) == 1,   'ERROR - offer must have exactly one offer_type'


    def transcript(self, recipient_id):
        trs = {'time': self.timestamp,
               'person': recipient_id,
               'event': 'offer received',
               'value': {
                   'offer id': self.id
               }}

        return json.dumps(trs)


    def viewed_offer_transcript(self, world, recipient_id):
        trs = {'time': world.world_time,
               'person': recipient_id,
               'event': 'offer viewed',
               'value': {
                   'offer id': self.id
               }}

        return json.dumps(trs)


    def offer_completed_transacript(self, world, recipient_id):
        trs = {'time': world.world_time,
               'person': recipient_id,
               'event': 'offer completed',
               'value': {
                   'offer_id': self.id,
                   'reward': self.reward}}

        return json.dumps(trs)


    def to_serializable(self):
        offer_dict = {'type':        self.type,
                      'timestamp':   self.timestamp,
                      'id':          self.id,
                      'valid_from':  self.valid_from,
                      'valid_until': self.valid_until,
                      'difficulty':  self.difficulty,
                      'progress':    self.progress,
                      'completed':   self.completed,
                      'reward':      self.reward,
                      'channel':     self.channel.to_serializable(),
                      'offer_type':  self.offer_type.to_serializable()
                      }

        return offer_dict


    @staticmethod
    def from_dict(offer_dict):
        assert offer_dict.get('type') == 'offer', 'ERROR - Dictionary must assert that it represents an Offer, but type is {}.'.format(offer_dict.get('type'))

        offer = Offer(            offer_dict.get('timestamp'),
                                  id         =offer_dict.get('id'),
                                  valid_from =offer_dict.get('valid_from'),
                                  valid_until=offer_dict.get('valid_until'),
                                  difficulty =offer_dict.get('difficulty'),
                                  reward     =offer_dict.get('reward'),
                                  progress   =offer_dict.get('progress'),
                                  completed  =offer_dict.get('completed'),
                                  channel    =Categorical.from_dict(offer_dict.get('channel')),
                                  offer_type =Categorical.from_dict(offer_dict.get('offer_type'))
                                  )
        return offer


    @staticmethod
    def from_json(json_string):
        offer_dict = json.loads(json_string)
        offer = Offer.from_dict(offer_dict)

        return offer


    @staticmethod
    def print_help():
        offer = Offer(0)

        print 'timestamp: int between {} and {}'.format(Constants.BEGINNING_OF_TIME, Constants.END_OF_TIME)
        print 'valid_from: int between {} and {}'.format(Constants.BEGINNING_OF_TIME, Constants.END_OF_TIME)
        print 'valid_to: int between {} and {}'.format(Constants.BEGINNING_OF_TIME, Constants.END_OF_TIME)
        print 'difficulty: positive numeric'
        print 'reward: positive numeric'
        print 'channel: Categorical({})'.format(offer.channel.names)
        print 'type: Categorical({})'.format(offer.offer_type.names)


    def is_active(self, current_time):
        """Determine if the offer is valid at the current time."""
        return True if self.valid_from <= current_time <= self.valid_until else False


class TestOffer(unittest.TestCase):
    """Test class for Offer."""

    def setUp(self):
        timestamp = 12345

        offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (0, 1, 1, 1))
        offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 0, 1))

        self.offer = Offer(timestamp, channel=offer_channel, offer_type=offer_type)


    def test_print_help(self):
        self.offer.print_help()
        self.assertTrue(1)


    def test_serializaton(self):
        offer_dict = self.offer.to_serializable()
        offer_reconstituted = Offer.from_dict(offer_dict)
        offer_reconstituted_dict = offer_reconstituted.to_serializable()
        self.assertTrue(offer_reconstituted_dict == offer_dict)

        offer_json = self.offer.to_json()
        offer_reconstituted = Offer.from_json(offer_json)
        offer_reconstituted_dict = offer_reconstituted.to_serializable()
        self.assertTrue(offer_reconstituted_dict == offer_dict)


class Transaction(Event):
    """Offer events occur when a Person receives an offer and offer an incentive to make a purchase.

    timestamp: the time the offer is received
    amount: the amount of the purchase
    """

    def __init__(self, timestamp_received, **kwargs):
        valid_kwargs = set(('id', 'amount'))
        kwargs_name_set = set(kwargs.keys())
        assert kwargs_name_set.issubset(valid_kwargs), 'ERROR - Invalid kwargs: {}'.format(kwargs_name_set.difference(valid_kwargs))

        self.type = 'transaction'
        self.id = kwargs.get('id') if kwargs.get('id') is not None else uuid.uuid4().hex
        self.timestamp = timestamp_received
        self.amount = kwargs.get('amount', None)


    def transcript(self, purchaser_id):
        trs = {'time': self.timestamp,
               'person': purchaser_id,
               'event': 'transaction',
               'value': {
                   'amount': self.amount
               }}

        return json.dumps(trs)


    def to_serializable(self):
        """Returna serializable dictionary representation of this Transaction."""
        trx_dict = {'type':      self.type,
                    'timestamp': self.timestamp,
                    'id':        self.id,
                    'amount':    self.amount
                    }

        return trx_dict


    @staticmethod
    def from_dict(trx_dict):
        """Create a Transaction Event from a dictionary."""
        assert trx_dict.get('type') == 'transaction', 'ERROR - Dictionary must assert that it represents a Transaction, but type is {}.'.format(trx_dict.get('type'))

        trx = Transaction(        trx_dict.get('timestamp'), \
                                  id     =trx_dict.get('id'), \
                                  amount =trx_dict.get('amount'))
        return trx


    @staticmethod
    def from_json(json_string):
        trx_dict = json.loads(json_string)
        trx = Transaction.from_dict(trx_dict)

        return trx


class TestTransaction(unittest.TestCase):
    """Test class for Transaction."""

    def setUp(self):
        timestamp = 12345
        amount = 1.0

        self.transaction = Transaction(timestamp, amount=amount)


    def test_serializaton(self):
        trx_dict = self.transaction.to_serializable()
        trx_reconstituted = Transaction.from_dict(trx_dict)
        trx_reconstituted_dict = trx_reconstituted.to_serializable()
        self.assertTrue(trx_reconstituted_dict == trx_dict)

        trx_json = self.transaction.to_json()
        trx_reconstituted = Transaction.from_json(trx_json)
        trx_reconstituted_dict = trx_reconstituted.to_serializable()
        self.assertTrue(trx_reconstituted_dict == trx_dict)


