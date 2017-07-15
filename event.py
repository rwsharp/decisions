"""Events are the actions and interactions that occur with respect to a Person.

    A Person's history is a timeseries of Events combined with a timeseries of attribute changes. We treat them
    separately, instead of creating a "state_change" event, because state_changes are intrinsic to the individual,
    whereas events are extrinsic.
"""

import logging
import unittest

import uuid
import numpy as np


BEGINNING_OF_TIME = 0
END_OF_TIME = 999999999


class Categorical(object):
    """A categorical variable class.

    For example, a Person can have weighted membership in each category of a segment variable. The names and weights are
    represented by separate numpy arrays. These variables are often used in a one-hot encoding context, so by using
    separate name and weight arrays instead of a single dictionary, we can do linear algebra with teh weights without
    the overhead of pulling values out of a dictionary in a specified order.
    """
    names = np.array(tuple())
    weights = np.array(tuple())

    def __init__(self, names=None, weights=None):
        """Initialize Segment."""

        if names is not None:
            assert len(set(names)) == len(names), 'ERROR - Not all names are unique.'
            self.names = np.array(names)

            if weights is None:
                self.weights = np.ones(len(names), dtype=np.int)
            else:
                assert len(names) == len(weights), 'ERROR - The number of names does not match the number of weights.'
                self.weights = np.array(weights)


    def get(self, name, default=None):
        """Indexing by segment name."""
        locations = np.where(self.names == name)[0]
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
        locations = np.where(self.names == name)[0]
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

        self.weights = np.array([self.get(name) for name in names])
        self.names = np.array(names)



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

    def __init__(self, timestamp, **kwargs):
        self.timestamp = timestamp
        self.valid_from = kwargs.get('valid_from', BEGINNING_OF_TIME)
        self.valid_until = kwargs.get('valid_until', END_OF_TIME)
        self.difficulty = kwargs.get('difficulty', 0)
        self.reward = kwargs.get('reward', 0)

        x = kwargs.get('channel')
        if x is not None:
            self.channel.set_equal(x)

        x = kwargs.get('type')
        if x is not None:
            self.type.set_equal(x)

        assert np.sum(self.channel.weights) > 0, 'ERROR - offer must have at least one channel'
        assert np.sum(self.type.weights) == 1,   'ERROR - offer must have exactly one type'



class TestEvent(unittest.TestCase):
    """Test class for Event."""

    def setUp(self):
        timestamp = 12345
        self.event = Event(timestamp)


    def test_init(self):
        print self.event.__dict__
        self.assertTrue(self.event)


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

        a = np.concatenate((np.array((0,)), x.weights))
        b = np.concatenate((np.array((0,)), y.weights))
        c = np.concatenate((np.array((0,)), z.weights))

        new_names = np.concatenate((np.array(('null',)), x.names))

        prod_ab = a * b

        r = Categorical(names=new_names, weights=prod_ab)
        s = Categorical(names=new_names, weights=c)

        self.assertTrue(r.compare_strict_equality(s))
        self.assertTrue(np.sum(r.weights) == 10)

