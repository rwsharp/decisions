"""Person

A Person is defined by behavioral characteristics and a state (the history of all offers received and actions taken by
the individual). For convenience, the current state (a snapshot of the history using the most recent values) is also
represesnted.
"""

#todo: people forget. truncate the history at n days. long-term memory is achieved by shifting the baseline behavior.

import logging
import unittest

import uuid
import numpy as np

class Person():
    """Represent an individual and simulate the person's actions."""
    channel_names = ('email', 'app', 'web')
    n_channels = len(channel_names)

    marketing_segment_names  = ('front page', 'local', 'entertainment', 'sports', 'opinion', 'comics')
    n_marketing_segments = len(marketing_segment_names)

    def __init__(self):
        """Initialize Person.

        Args:
        """
        raise NotImplementedError()

        self.attributes = {'id': uuid.uuid4(),
                           'age': None,
                           'gender': None,
                           'tenure': None,
                           'income': None,
                           'channel': Segment(channel_names),
                           'marketing segment': Segment(marketing_segment_names)}

        self.history = {} # timeseries dictionary of transactions and offers key = simtime timestamp, val = {type: 'trx', value: '1.50'}

        self.state = {
            'time since last trx': None,
            'active offers': None,
            'last offer viewed': None
        }

        logging.info('Person initialized')


    def simulate(self):
        """Simulate the actions of an individual."""
        raise NotImplementedError()


class Segment():
    """A segment variable class. People have weighted membership in each category of a segment variable."""
    def __init__(self, names, weights=None):
        """Initialize Segment."""
        self.names = names
        if weights is None:
            self.weights = np.ones(len(names), dtype=np.int)


class TestPerson(unittest.TestCase):
    """Test class for Person."""

    def setUp(self):
        self.Person = Person()


class TestSegment(unittest.TestCase):
    """Test class for Segment."""

    def setUp(self):
        self.Segment = Segment()