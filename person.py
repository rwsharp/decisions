"""Person

A Person is defined by behavioral characteristics and a state (the history of all offers received and actions taken by
the individual). For convenience, the current state (a snapshot of the history using the most recent values) is also
represesnted.
"""

# todo: people forget. truncate the history at n days. long-term memory is achieved by shifting the baseline behavior.

import logging
import unittest

import uuid
import numpy

from externalities import Constants, Categorical, Event, Offer, Transaction, World


class Person(object):
    """Represent an individual and simulate the person's actions."""

    taste_names = ('sweet', 'sour', 'salty', 'bitter', 'umami')
    channel_names = Offer.channel.names
    marketing_segment_names = ('front page', 'local', 'entertainment', 'sports', 'opinion', 'comics')

    # outlier_frequency = n * 0.0001 means that n in every 10,000 purchases is a big one
    outlier_frequency = 0.0005

    def __init__(self, became_member_on, **kwargs):
        """Initialize Person.

        became_member_on: date (cannot be missing - assigned when an individual becomes a member)

        kwargs:
            dob: date (default = 19010101 - sound silly? it happens in the real world and skews age distributions)
            gender: M, F, O (O = other, e.g., decline to state, does not identify, etc.)
            income: positive int, None
            taste: categorical('sweet', 'sour', 'salty', 'bitter', 'umami')
            channel: categorical('email', 'app', 'web')
            marketing_segment: categorical('front page', 'local', 'entertainment', 'sports', 'opinion', 'comics')
            offer_sensitivity: categorical
            make_purchase_sensitivity: ???
        """

        ######################
        # Intrinsic Attributes
        ######################

        self.id = uuid.uuid4()
        self.dob = kwargs.get('dob', '19010101') # set a sneaky default value for unknown
        self.gender = kwargs.get('gender', None)
        self.became_member_on = became_member_on
        self.income = kwargs.get('income', None)

        default_taste = Categorical(self.taste_names)
        kwargs_taste = kwargs.get('taste', None)
        if kwargs_taste is not None:
            assert default_taste.same_names(kwargs_taste), 'ERROR - keyword argument taste must have names = {}'.format(default_taste.names)
            self.taste = kwargs_taste
            self.taste.set_order(default_taste.names)
        else:
            self.taste = default_taste

        default_channel = Categorical(self.channel_names)
        kwargs_channel = kwargs.get('channel', None)
        if kwargs_taste is not None:
            assert default_channel.same_names(kwargs_channel), 'ERROR - keyword argument channel must have names = {}'.format(default_channel.names)
            self.channel = kwargs_channel
            self.channel.set_order(default_channel)
        else:
            self.channel = default_channel

        default_marketing_segment = Categorical(self.marketing_segment_names)
        kwargs_marketing_segment = kwargs.get('marketing_segment')
        if kwargs_marketing_segment is not None:
            assert default_marketing_segment.same_names(kwargs_marketing_segment), 'ERROR - keyword argument marketing_segment must have names = {}'.format(default_marketing_segment.names)
            self.marketing_segment = kwargs_marketing_segment
            self.marketing_segment.set_order(kwargs_marketing_segment)
        else:
            self.marketing_segment = default_marketing_segment

        ######################
        # Extrinsic Attributes
        ######################

        self.last_transaction = None
        self.last_unviewed_offer = None
        self.last_viewed_offer = None
        # only allow one offer to be active at a time, and only count as active if it's been viewed (no accidental
        # winners) - if offers have overlapping validity periods, then last received is the winner
        self.last_viewed_offer_active = None

        #########
        # History
        #########

        # A list of events. Since events have a timestamp, this is equivalent to a time series.
        self.history = list()

        ###################
        # Sensitivity
        ###################

        # view_offer_sensitivity
        view_offer_sensitivity_names = numpy.concatenate((numpy.array(('offer_age',)), self.channel_names))
        default_view_offer_sensitivity = Categorical(view_offer_sensitivity_names)
        default_view_offer_sensitivity.set('offer_age', -1)
        kwargs_view_offer_sensitivity = kwargs.get('view_offer_sensitivity', None)
        if kwargs_view_offer_sensitivity is not None:
            assert default_view_offer_sensitivity.same_names(kwargs_view_offer_sensitivity), 'ERROR - keyword argument view_offer_sensitivity must have names = {}'.format(default_view_offer_sensitivity.names)
            self.view_offer_sensitivity = kwargs_view_offer_sensitivity
            self.view_offer_sensitivity.set_order(default_view_offer_sensitivity.names)
        else:
            self.view_offer_sensitivity = default_view_offer_sensitivity

        # make_purchase_sensitivity
        make_purchase_sensitivity_names = numpy.concatenate((numpy.array(('time_since_last_transaction',
                                                                          'time_since_last_viewed_offer',
                                                                          'viewed_active_offer')),
                                                             self.channel_names))
        default_make_purchase_sensitivity = Categorical(make_purchase_sensitivity_names)
        default_make_purchase_sensitivity.set('time_since_last_viewed_offer', -1)
        kwargs_make_purchase_sensitivity = kwargs.get('make_purchase_sensitivity', None)
        if kwargs_make_purchase_sensitivity is not None:
            assert default_make_purchase_sensitivity.same_names(kwargs_make_purchase_sensitivity), 'ERROR - keyword argument make_purchase_sensitivity must have names = {}'.format(default_make_purchase_sensitivity.names)
            self.make_purchase_sensitivity = kwargs_make_purchase_sensitivity
            self.make_purchase_sensitivity.set_order(default_make_purchase_sensitivity.names)
        else:
            self.make_purchase_sensitivity = default_make_purchase_sensitivity

        # purchase_amount_sensitivity
        purchase_amount_sensitivity_names = numpy.concatenate((numpy.array(('constant',
                                                                            'income_adjusted_purchase_sensitivity')),
                                                               self.marketing_segment_names,
                                                               self.taste_names))
        default_purchase_amount_sensitivity = Categorical(purchase_amount_sensitivity_names)
        kwargs_purchase_amount_sensitivity = kwargs.get('purchase_amount_sensitivity', None)
        if kwargs_purchase_amount_sensitivity is not None:
            assert default_purchase_amount_sensitivity.same_names(kwargs_purchase_amount_sensitivity), 'ERROR - keyword argument purchase_amount_sensitivity must have names = {}'.format(default_purchase_amount_sensitivity.names)
            default_purchase_amount_sensitivity.set('income_adjusted_purchase_sensitivity', 1)
            self.purchase_amount_sensitivity = kwargs_purchase_amount_sensitivity
            self.purchase_amount_sensitivity.set_order(default_purchase_amount_sensitivity.names)
        else:
            self.purchase_amount_sensitivity = default_purchase_amount_sensitivity

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

        viewed_offer_event = self.view_offer(current_time)
        if viewed_offer_event:
            logging.DEBUG('Viewed offer at {}: {}'.format(current_time, self.last_viewed_offer.__dict__))

        purchase_event = self.make_purchase(current_time)
        if purchase_event:
            logging.DEBUG('Made purchase at {}: {}'.format(current_time, self.last_transaction.__dict__))


    def view_offer(self, current_time):
        offer = self.last_unviewed_offer
        if offer is not None:
            offer_age = current_time - offer.timestamp
        else:
            # there is no offer to view, we're done here
            return None

        beta = self.offer_sensitivity
        x    = numpy.concatenate(numpy.array((offer_age,)), offer.channel.weights)
        p = 1.0 / (1.0 + numpy.exp(-numpy.sum(beta * x)))

        # flip a coin to decide if the offer was viewed
        viewed_offer = True if numpy.random.random() < p else False

        if viewed_offer:
            offer_viewed = self.last_unviewed_offer
            self.last_viewed_offers = offer_viewed
            # by setting the last unviewed offer to None here, we're assuming that only the most recently
            # received offer will every be viewed. Once it's not at the top of the inbox, it's forgotten.
            self.last_unviewed_offer = None
        else:
            offer_viewed = None

        return offer_viewed


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
            time_since_last_viewed_offer = Constants.END_OF_TIME - Constants.BEGINNING_OF_TIME

        # Is last viewed offer active?
        viewed_active_offer = 1 if self.last_viewed_offer.is_active(current_time) else 0

        beta = self.make_purchase_sensitivity
        x    = numpy.concatenate((numpy.array((time_since_last_transaction,
                                               time_since_last_viewed_offer,
                                               viewed_active_offer)),
                                  offer.channel.weights))
        p = 1.0 / (1.0 + numpy.exp(-numpy.sum(beta * x)))

        # flip a coin to decide if a purchase was made
        made_purchase = True if numpy.random.random() < p else False

        if made_purchase:
            # Determine if this is an outlier order or regular order
            if numpy.random.random() < self.outlier_frequency:
                purchase_amount = self.outlier_purchase_amount()
            else:
                purchase_amount = self.purchase_amount()

            transaction = Transaction(timestamp=current_time, amount=purchase_amount)
            self.last_transaction = transaction
        else:
            transaction = None

        return transaction


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
        x = numpy.concatenate((numpy.array((1,
                                            income_adjusted_purchase_sensitivity)),
                               self.marketing_segment.weights,
                               self.taste.weights))

        mean = numpy.sum(beta * x)

        # simple relationship between mean and var reflects the increased options for purchase to achieve higher mean
        var = 2.0 * mean

        # mean = k*theta
        # var = k*theta**2

        theta = var / mean
        k = mean ** 2 / var

        # minimum purchase is $0.05
        # all purchases are rounded to a whole number of cents
        z = max(0.05, round(numpy.random.gamma(shape=k, scale=theta, size=None), 2))

        # todo: add outliers from a different distribution

    def outlier_purchase_amount(self, current_time):
        """Randomly sample an outlier distribution to determine the amount of a purchase."""
        raise NotImplementedError

class TestPerson(unittest.TestCase):
    """Test class for Person."""

    def setUp(self):
        self.person = Person(became_member_on=12345)


    def test_person(self):
        self.assertTrue(self.person)


