"""Person

A Person is defined by behavioral characteristics and a state (the history of all offers received and actions taken by
the individual). For convenience, the current state (a snapshot of the history using the most recent values) is also
represesnted.
"""

# todo: people forget. truncate the history at n days. long-term memory is achieved by shifting the baseline behavior.

import logging
import unittest

import uuid
import numpy as np

from event import *


class Person(object):
    """Represent an individual and simulate the person's actions."""
    taste = Categorical(('sweet', 'sour', 'salty', 'bitter', 'umami'))
    channel = Categorical(('email', 'app', 'web'))
    marketing_segment = Categorical(('front page', 'local', 'entertainment', 'sports', 'opinion', 'comics'))

    # outlier_frequency = n * 0.0001 means that n in every 10,000 purchases is a big one
    outlier_frequency = 0.0005

    def __init__(self, taste, channel, marketing_segment, offer_sensitivity, make_purchase_sensitivity):
        """Initialize Person.

        Args:
        """

        ######################
        # Intrinsic Attributes
        ######################

        self.id = uuid.uuid4()
        self.dob = None
        self.gender = None
        self.became_member_on = None
        self.income = None
        self.taste.set_equal(taste)
        self.channel.set_equal(channel)
        self.marketing_segment.set_equal(marketing_segment)

        ######################
        # Extrinsic Attributes
        ######################

        self.last_transaction = None
        self.last_unviewed_offer = None
        self.last_viewed_offer = None
        self.active_viewed_offer = None # only allow one offer to be active at a time, and only count as active if it's been viewed (no accidental winners) - if offers have overlapping validity periods, then last received is the winner

        #########
        # History
        #########

        self.history = list()

        ###################
        # Offer Sensitivity
        ###################

        self.offer_sensitivity = Categorical(names=None, weights=None).set_equal(offer_sensitivity)
        self.make_purchase_sensitivity = Categorical(names=None, weights=None).set_equal(make_purchase_sensitivity)

        logging.info('Person initialized')


    def update_state(self, current_time):
        """Update customer state variables.

        The customer is only influenced by the last viewed offer (older offers are forgotten). If the last viewed offer
        is currently active, update the active viewed offer.
        """

        # If the active viewed offer is no longer valid, reset it.
        offer = self.active_viewed_offer
        if offer is not None:
            if current_time > offer.valid_until:
                self.active_viewed_offer = None

        # Check if the last viewed offer has become active
        offer = self.last_viewed_offer
        if offer is not None:
            if offer.valid_from <= current_time <= offer.valid_until:
                self.active_viewed_offer = offer


    def simulate(self, current_time):
        """Simulate the actions of an individual.

        There actions a person can take are:
            * make a purchase
            * view an offer

        There is no explicit choice to win an offer. Instead, an offer has an associated difficulty, which is a dollar
        amount. If the person spends this amount in the validity period, the offer is won (and the promoter must pay
        out an amount equal to the reward).

        Making a purchase consists of first choosing whether or not to make a purchase at all, and then deciding what to
        spend. There are no explicit items for sale, just an amount spent.

        Choosing whether depends on: time of day, segment, income, how long since last purchase, offers
        - it's a binary choice, so use logistic function

        the simulation is for a single hour wth the given start_time
        each hour we compute the probability that a purchase will occur, then flip a coin to decide if it does
        the timestamp on the purchase is the date + hour at the end of the step in which the purchase occurs
        if it does, we flip more coins to decide what

        probability that it will occur is logit()
        # todo: bursts?

        Choosing how much to spend depends on: segment, income, taste, offer ...
        the amount to spend is a gamma parameterized by mean and variance
        the mean and variance themselves are linear combinations of segment, income, taste, offer, ...
        in this way, an offer can excite purchase amount by increasing mean spend

        deciding if an offer was viewed is a binary choice that we ask until it is viewed
        the probability decreases with time
        """


        viewed_offer = self.view_offer(current_time)
        if viewed_offer:
            logging.DEBUG('Viewed offer at {}: {}'.format(current_time, self.last_viewed_offer.__dict__))

        made_purchase, purchase_amount = self.make_purchase(current_time)
        if made_purchase:
            logging.DEBUG('Made purchase at {}: {}'.format(current_time, purchase_amount))

        raise NotImplementedError()


    def view_offer(self, current_time):
        offer = self.last_unviewed_offer
        if offer is not None:
            offer_age = current_time - offer.timestamp
        else:
            offer_age = 0

        beta = self.offer_sensitivity
        x    = np.concatenate(np.array((offer_age,)), offer.channel.weights)
        p = 1.0 / (1.0 + np.exp(-np.sum(beta * x)))

        # flip a coin to decide if the offer was viewed
        viewed_offer = True if np.random.random() < p else False

        if viewed_offer:
            self.last_viewed_offers = self.last_unviewed_offer
            self.last_unviewed_offer = None

            # by setting the last viewed offer to None here, we're assuming that offers that only the most recently
            # received offer will every be viewed. Once it's not at the top of the inbox, it's forgotten.

        return viewed_offer


    def make_purchase(self, current_time):
        """Person decides whether to make a purcahse or not and the size of the purchase. Includes outliers, e.g., due
        to large group orders vs. individual orders.

        Depends on time of day, segment, income, how long since last purchase, offers
        """

        # How long since last transaction
        if self.last_transaction is not None:
            time_since_last_transaction = current_time - self.last_transaction.timestamp
        else:
            time_since_last_transaction = 0

        # How long since last viewed offer
        offer = self.last_viewed_offer
        if offer is not None:
            time_since_last_viewed_offer = current_time - offer.timestamp
        else:
            # never viewed an offer, so as if it's been forever
            time_since_last_viewed_offer = END_OF_TIME - BEGINNING_OF_TIME

        # Is last viewed offer active?
        offer = self.viewed_active_offer
        if offer is not None:
            viewed_active_offer = 1
        else:
            viewed_active_offer = 0

        beta = self.make_purchase_sensitivity
        x    = np.concatenate((np.array((time_since_last_transaction,
                                         time_since_last_viewed_offer,
                                         viewed_active_offer)),
                               offer.channel.weights))
        p = 1.0 / (1.0 + np.exp(-np.sum(beta * x)))

        # flip a coin to decide if a purchase was made
        made_purchase = True if np.random.random() < p else False

        if made_purchase:
            # Determine if this is an outlier order or regular order
            if np.random.random() < self.outlier_frequency:
                purchase_amount = self.outlier_purchase_amount()
            else:
                purchase_amount = self.purchase_amount()
        else:
            purchase_amount = None

        return (made_purchase, purchase_amount)


    def purchase_amount(self, current_time):
        """Randomly sample a gamma distribution with special outliers determine the amount of a purchase.

        Mean purchase amount depends on: average transaction amount, segment, income, taste, offer

        Order of components in sensitivity and value arrays (beta and x) is:
        0: average_transaction_amount (value = 1, this is the bias / constant term)
        1: income (sensitivity = 1, the relationship is nonlinear and captured by the special "income_curve" function)
        2: marketing_segment components
        3: taste components
        """

        # average purchase increases with income, but has a minimum (min price of a product) and tops out at some level:
        # linear with min and max plateaus at thresholds
        min_income = 10000.0
        max_income = 100000.0
        min_mean_purchase = 1.0
        max_mean_purchase = 25.0

        slope = (max_income - min_income)/(max_mean_purchase - min_mean_purchase)

        if self.income <= min_income:
            adjusted_income = 0.0
        elif self.income <= max_income:
            adjusted_income = self.income
        else:
            adjusted_income = max_income

        income_adjusted_purchase_sensitivity = min_mean_purchase + slope*adjusted_income

        beta = self.purchase_amount_sensitivity
        x = np.concatenate((np.array((1,
                                      income_adjusted_purchase_sensitivity)),
                            self.marketing_segment.weights,
                            self.taste.weights))

        mean = np.sum(beta * x)

        # simple relationship between mean and var reflects the increased options for purchase to achieve higher mean
        var = 2.0 * mean

        # mean = k*theta
        # var = k*theta**2

        theta = var / mean
        k = mean ** 2 / var

        # minimum purchase is $0.05
        # all purchases are rounded to a whole number of cents
        z = max(0.05, round(np.random.gamma(shape=k, scale=theta, size=None), 2))

        # todo: add outliers from a different distribution

    def outlier_purchase_amount(self, current_time):
        """Randomly sample an outlier distribution to determine the amount of a purchase."""
        raise NotImplementedError

class TestPerson(unittest.TestCase):
    """Test class for Person."""

    def setUp(self):
        self.Person = Person()


