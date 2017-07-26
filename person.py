"""Person

A Person is defined by behavioral characteristics and a state (the history of all offers received and actions taken by
the individual). For convenience, the current state (a snapshot of the history using the most recent values) is also
represesnted.
"""

# todo: people forget. truncate the history at n days. long-term memory is achieved by shifting the baseline behavior.

import logging
import unittest

import copy
import json
import datetime

import uuid
import numpy

from externalities import Constants, Categorical, Event, Offer, Transaction, World


class Person(object):
    """Represent an individual and simulate the person's actions."""

    taste_names = ('sweet', 'sour', 'salty', 'bitter', 'umami')
    marketing_segment_names = ('front page', 'local', 'entertainment', 'sports', 'opinion', 'comics')

    # outlier_frequency = n * 0.001 means that n in every 1,000 purchases is a big one
    outlier_frequency = 0.005


    @staticmethod
    def bounded_response(x, min_x, max_x, f_of_min_x, f_of_max_x):
        """Bounded piecewise linear response to a stimulus.

        Let x represent the independent variable and y the dependent variable. Given bounds on x and y, this function
        defines a response, f(x) that equals f_of_min_x when x <= min_x, f_of_max_x when x >= max_x, and the linear
        interpolant in the region between min_x and max_x.
        """

        slope = (f_of_max_x - f_of_min_x)/float(max_x - min_x)

        if x <= min_x:
            return_value = f_of_min_x
        elif x <= max_x:
            return_value = f_of_min_x + slope*(x - min_x)
        else:
            return_value = f_of_max_x

        return return_value


    @staticmethod
    def print_help():
        person = Person('20170716')

        print 'id: uuid.uuid4() (set automatically)'
        print 'dob: \'YYYYMMDD\''
        print 'gender: [\'M\', \'F\', \'O\']'
        print 'became_member_on: \'YYYYMMDD\''
        print 'income: positive numeric'
        print 'taste: Categorical([{}])'.format(','.join(map(lambda s: "'" + s + "'", person.taste.names)))
        print 'marketing_segment: Categorical([{}])'.format(','.join(map(lambda s: "'" + s + "'", person.marketing_segment.names)))
        print
        print 'last_transaction: Transaction'
        print 'last_unviewed_offer: Offer'
        print 'last_viewed_offer: Offer'
        print 'history: list of Offers and Transactions sorted by Event timestamps'
        print
        print 'view_offer_sensitivity: Categorical([{}])'.format(','.join(map(lambda s: "'" + s + "'", person.view_offer_sensitivity.names)))
        print 'make_purchase_sensitivity: Categorical([{}])'.format(','.join(map(lambda s: "'" + s + "'", person.make_purchase_sensitivity.names)))
        print 'purchase_amount_sensitivity: Categorical([{}])'.format(','.join(map(lambda s: "'" + s + "'", person.purchase_amount_sensitivity.names)))


    def __init__(self, became_member_on, **kwargs):
        """Initialize Person.

        became_member_on: date (cannot be missing - assigned when an individual becomes a member)

        kwargs:
            dob: date (default = 19010101 - sound silly? it happens in the real world and skews age distributions)
            gender: M, F, O (O = other, e.g., decline to state, does not identify, etc.)
            income: positive int, None
            taste: categorical('sweet', 'sour', 'salty', 'bitter', 'umami')
            marketing_segment: categorical('front page', 'local', 'entertainment', 'sports', 'opinion', 'comics')
            offer_sensitivity: categorical
            make_purchase_sensitivity: ???
        """
        valid_kwargs = {'id',
                        'dob',
                        'gender',
                        'income',
                        'taste',
                        'marketing_segment',
                        'last_transaction',
                        'last_unviewed_offer',
                        'last_viewed_offer',
                        'history',
                        'view_offer_sensitivity',
                        'make_purchase_sensitivity',
                        'purchase_amount_sensitivity'
                        }
        kwargs_name_set = set(kwargs.keys())
        assert kwargs_name_set.issubset(valid_kwargs), 'ERROR - Invalid kwargs: {}'.format(kwargs_name_set.difference(valid_kwargs))

        ######################
        # Intrinsic Attributes
        ######################

        self.id =     kwargs.get('id') if kwargs.get('id') is not None else uuid.uuid4().hex
        self.dt_fmt = '%Y%m%d'
        try:
            datetime.datetime.strptime(kwargs.get('dob'), self.dt_fmt)
            self.dob = kwargs.get('dob')
        except:
            self.dob = '19010101'
        self.gender = kwargs.get('gender')
        try:
            datetime.datetime.strptime(became_member_on, self.dt_fmt)
            self.became_member_on = became_member_on
        except:
            raise ValueError('ERROR - became_member_on has invalid format (should be: {}). became_member_on={}'.format(self.dt_fmt, became_member_on))
        self.income = kwargs.get('income')

        default_taste = Categorical(self.taste_names)
        kwargs_taste = kwargs.get('taste')
        if kwargs_taste is not None:
            assert default_taste.compare_names(kwargs_taste), 'ERROR - keyword argument taste must have names = {}'.format(default_taste.names)
            self.taste = kwargs_taste
            self.taste.set_order(default_taste.names)
        else:
            self.taste = default_taste

        default_marketing_segment = Categorical(self.marketing_segment_names)
        kwargs_marketing_segment = kwargs.get('marketing_segment')
        if kwargs_marketing_segment is not None:
            assert default_marketing_segment.compare_names(kwargs_marketing_segment), 'ERROR - keyword argument marketing_segment must have names = {}'.format(default_marketing_segment.names)
            self.marketing_segment = kwargs_marketing_segment
            self.marketing_segment.set_order(kwargs_marketing_segment.names)
        else:
            self.marketing_segment = default_marketing_segment

        ######################
        # Extrinsic Attributes
        ######################

        # only allow one offer to be active at a time, and only count as active if it's been viewed (no accidental
        # winners) - if offers have overlapping validity periods, then last received is the winner

        # Person has a short memory. Only the most recently received offer can be viewed (a newly received offer will
        # supplant it), and only the most recently viewed offer can influence Person's behavior (view another and Person
        # forgets). However, whether an offer is viewed or not, the user can still accidentally win by making a
        # sufficient purchase. If two offers are open simultanrously, then Person can get double credit (win both) with
        # a single purchase.

        self.last_transaction    = kwargs.get('last_transaction')
        self.last_unviewed_offer = kwargs.get('last_unviewed_offer')
        self.last_viewed_offer   = kwargs.get('last_viewed_offer')

        #########
        # History
        #########

        # A list of events. Since events have a timestamp, this is equivalent to a time series.
        kwargs_history = kwargs.get('history', list())
        # note that all(list()) returns True
        assert all(map(lambda e: isinstance(e, Event), kwargs_history)), 'ERROR - Not all items in history are of type Event.'
        self.history = kwargs_history

        ###################
        # Sensitivity
        ###################

        # view_offer_sensitivity
        view_offer_sensitivity_names = numpy.concatenate((numpy.array(('background', 'offer_age')), Offer(0).channel.names))
        default_view_offer_sensitivity = Categorical(view_offer_sensitivity_names)
        default_view_offer_sensitivity.set('offer_age', -1)
        kwargs_view_offer_sensitivity = kwargs.get('view_offer_sensitivity', None)
        if kwargs_view_offer_sensitivity is not None:
            assert default_view_offer_sensitivity.compare_names(kwargs_view_offer_sensitivity), 'ERROR - keyword argument view_offer_sensitivity must have names = {}'.format(default_view_offer_sensitivity.names)
            self.view_offer_sensitivity = copy.deepcopy(kwargs_view_offer_sensitivity)
            self.view_offer_sensitivity.set_order(default_view_offer_sensitivity.names)
        else:
            self.view_offer_sensitivity = default_view_offer_sensitivity

        # make_purchase_sensitivity
        make_purchase_sensitivity_names = numpy.array(('background',
                                                       'time_since_last_transaction',
                                                       'last_viewed_offer_strength',
                                                       'viewed_active_offer'))
        default_make_purchase_sensitivity = Categorical(make_purchase_sensitivity_names)
        default_make_purchase_sensitivity.set('time_since_last_viewed_offer', -1)
        kwargs_make_purchase_sensitivity = kwargs.get('make_purchase_sensitivity', None)
        if kwargs_make_purchase_sensitivity is not None:
            assert default_make_purchase_sensitivity.compare_names(kwargs_make_purchase_sensitivity), 'ERROR - keyword argument make_purchase_sensitivity must have names = {}'.format(default_make_purchase_sensitivity.names)
            self.make_purchase_sensitivity = copy.deepcopy(kwargs_make_purchase_sensitivity)
            self.make_purchase_sensitivity.set_order(default_make_purchase_sensitivity.names)
        else:
            self.make_purchase_sensitivity = default_make_purchase_sensitivity

        # purchase_amount_sensitivity
        purchase_amount_sensitivity_names = numpy.concatenate((numpy.array(('background',
                                                                            'income_adjusted_purchase_sensitivity')),
                                                               self.marketing_segment_names,
                                                               self.taste_names))
        default_purchase_amount_sensitivity = Categorical(purchase_amount_sensitivity_names)
        kwargs_purchase_amount_sensitivity = kwargs.get('purchase_amount_sensitivity', None)
        if kwargs_purchase_amount_sensitivity is not None:
            assert default_purchase_amount_sensitivity.compare_names(kwargs_purchase_amount_sensitivity), 'ERROR - keyword argument purchase_amount_sensitivity must have names = {}'.format(default_purchase_amount_sensitivity.names)
            default_purchase_amount_sensitivity.set('income_adjusted_purchase_sensitivity', 1)
            self.purchase_amount_sensitivity = copy.deepcopy(kwargs_purchase_amount_sensitivity)
            self.purchase_amount_sensitivity.set_order(default_purchase_amount_sensitivity.names)
        else:
            self.purchase_amount_sensitivity = default_purchase_amount_sensitivity

        logging.info('Person initialized')


    def to_serializable(self):
        """Create a serializable representation."""
        person_dict = {'id':                          self.id,
                       'became_member_on':            self.became_member_on,
                       'dob':                         self.dob,
                       'gender':                      self.gender,
                       'income':                      self.income,
                       'taste':                       self.taste.to_serializable(),
                       'marketing_segment':           self.marketing_segment.to_serializable(),
                       'last_transaction':            self.last_transaction,
                       'last_unviewed_offer':         self.last_unviewed_offer,
                       'last_viewed_offer':           self.last_viewed_offer,
                       'history':                     [event.to_serializable() for event in self.history],
                       'view_offer_sensitivity':      self.view_offer_sensitivity.to_serializable(),
                       'make_purchase_sensitivity':   self.make_purchase_sensitivity.to_serializable(),
                       'purchase_amount_sensitivity': self.purchase_amount_sensitivity.to_serializable()}

        return person_dict


    @staticmethod
    def from_dict(person_dict):

        history = list()
        for event_dict in person_dict.get('history'):
            event_type = event_dict.get('type')
            if event_type == 'event':
                event = Event.from_dict(event_dict)
            elif event_type == 'offer':
                event = Offer.from_dict(event_dict)
            elif event_type == 'transaction':
                event = Transaction.from_dict(event_dict)
            else:
                raise ValueError('ERROR - Event type not recognized ({}).'.format(event_type))

            history.append(event)

        person = Person(                            person_dict.get('became_member_on'),                                        \
                                                 id=person_dict.get('id'),                                                      \
                                                dob=person_dict.get('dob'),                                                     \
                                             gender=person_dict.get('gender'),                                                  \
                                             income=person_dict.get('income'),                                                  \
                                              taste=Categorical.from_dict(person_dict.get('taste')),                            \
                                  marketing_segment=Categorical.from_dict(person_dict.get('marketing_segment')),                \
                                   last_transaction=person_dict.get('last_transaction'),                                        \
                                last_unviewed_offer=person_dict.get('last_unviewed_offer'),                                     \
                                  last_viewed_offer=person_dict.get('last_viewed_offer'),                                       \
                                            history=history,                                                                    \
                             view_offer_sensitivity=Categorical.from_dict(person_dict.get('view_offer_sensitivity')),           \
                          make_purchase_sensitivity=Categorical.from_dict(person_dict.get('make_purchase_sensitivity')),        \
                        purchase_amount_sensitivity=Categorical.from_dict(person_dict.get('purchase_amount_sensitivity')))

        return person


    @staticmethod
    def from_json(json_string):
        person_dict = json.loads(json_string)
        person = Person.from_dict(person_dict)

        return person


    def to_json(self):
        """Create a json representation."""
        json_string = json.dumps(self.to_serializable())

        return json_string


    def update_offer_state(self, world, transaction):
        """Update state of offers when a customer makes a transaction.

        Determine which offers are open, whether they've been completed, and add transaction amount toward completion.
        """

        # Offer completion rules
        # BOGO: at least purchase in the validity period that is greated than difficulty
        # discount: cumulative purchases in the validity period are greater than or equal to difficulty
        # information: nothing to do, so cannot be completed, also no reward

        transcript_items = list()

        # we need to go through all events since an offer could be open ended
        for event in self.history:
            if isinstance(event, Offer):
                offer_type = event.offer_type
                if event.reward > 0:
                    if event.is_active(world.world_time):
                        # update offer progress
                        if offer_type.get('bogo') == 1:
                            event.progress = transaction.amount if transaction.amount > event.difficulty else 0.00
                        elif offer_type.get('discount') == 1:
                            event.progress += transaction.amount

                        # has the offer been completed?
                        if offer_type.get('informational') == 0:
                            if event.progress >= event.difficulty:
                                if not event.completed:
                                    event.completed = True
                                    transcript = event.offer_completed_transacript(world, self.id)
                                    transcript_items.append(transcript)

        return transcript_items


    def update(self, world):
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

        transcript_items = list()

        viewed_offer_event = self.view_offer(world)
        if viewed_offer_event:
            transcript_items.append(viewed_offer_event.viewed_offer_transcript(world, self.id))
            # logging.debug('Viewed offer at {}: {}'.format(world.world_time, self.last_viewed_offer.__dict__))

        purchase_event = self.make_purchase(world)
        if purchase_event:
            transcript_items.append(purchase_event.transcript(self.id))
            # logging.debug('Made purchase at {}: {}'.format(world.world_time, self.last_transaction.__dict__))

            transcript_items.extend(self.update_offer_state(world, purchase_event))

        return transcript_items


    def receive_offer(self, world, offer):
        """Receive an offer from an external source."""

        # set the receipt time
        received_time = world.world_time
        offer.timestamp = received_time

        # update history
        offer_copy = copy.deepcopy(offer)
        self.history.append(offer_copy)
        self.last_unviewed_offer = offer_copy

        transcript_items = list((offer.transcript(self.id),))
        logging.debug('{} received offer {} at {}'.format(self.id, offer.id, received_time))

        return transcript_items


    def view_offer(self, world):
        # logging.debug('View offer decision at time t = {}'.format(world.world_time))

        offer = self.last_unviewed_offer
        if offer is not None:
            offer_age = world.world_time - offer.timestamp
        else:
            # there is no offer to view, we're done here
            offer_viewed = None
            return offer_viewed

        beta = self.view_offer_sensitivity.weights
        x    = numpy.concatenate((numpy.array((1,
                                               offer_age)),
                                  offer.channel.weights))
        p = 1.0 / (1.0 + numpy.exp(-numpy.dot(beta, x)))

        # logging.debug('        beta = {}'.format(beta))
        # logging.debug('           x = {}'.format(x))
        # logging.debug('      beta*x = {}'.format(beta*x))
        # logging.debug('dot(beta, x) = {}'.format(numpy.dot(beta, x)))
        # logging.debug('           p = {}'.format(p))

        # flip a coin to decide if the offer was viewed
        viewed_offer = True if numpy.random.random() < p else False

        if viewed_offer:
            # logging.debug('Offer viewed')
            offer_viewed = self.last_unviewed_offer
            self.last_viewed_offer = offer_viewed
            # by setting the last unviewed offer to None here, we're assuming that only the most recently
            # received offer will every be viewed. Once it's not at the top of the inbox, it's forgotten.
            self.last_unviewed_offer = None
        else:
            offer_viewed = None

        return offer_viewed


    def make_purchase(self, world):
        """Person decides whether to make a purcahse or not and the size of the purchase. Includes outliers, e.g., due
        to large group orders vs. individual orders.

        Depends on time of day, segment, income, how long since last purchase, offers
        """

        # logging.debug('Made purchase decision at time t = {}'.format(world.world_time))

        # How long since last transaction
        if self.last_transaction is not None:
            time_since_last_transaction = world.world_time - self.last_transaction.timestamp
        else:
            time_since_last_transaction = 0

        # How long since last viewed offer
        offer = self.last_viewed_offer
        if offer is not None:
            time_since_last_viewed_offer = world.world_time - offer.timestamp
            last_viewed_offer_duration = offer.valid_until - offer.timestamp
            viewed_active_offer = 1 if offer.is_active(world.world_time) else 0
            offer_channel_weights = offer.channel.weights
        else:
            # never viewed an offer, so as if it's been forever
            time_since_last_viewed_offer = Constants.END_OF_TIME - Constants.BEGINNING_OF_TIME
            last_viewed_offer_duration = 0
            viewed_active_offer = 0
            offer_channel_weights = Offer(Constants.BEGINNING_OF_TIME).channel.zeros

        # as time since last offer increases, the effect should go to zero: x_max = T, f_of_x_max = 0
        # the offer view is most powerful immediately: x_min = 0, f_of_x_min = 1
        # therefore we have a function that should decrease from 1 to 0 as x increases from 0 to T
        # also, the sensitivity should be positive (the negative effect lies in the state variable)
        # T is the time at which the viewed offer no longer has an effect
        # let's make this 3 days after the offer expires = offer_length + 24/float(world.world_time_tick) * 3
        last_viewed_offer_strength = self.bounded_response(time_since_last_viewed_offer,
                                                           min_x=0,
                                                           max_x=last_viewed_offer_duration + 24/float(world.world_time_tick + 1)*3,
                                                           f_of_min_x=1.0,
                                                           f_of_max_x=0.0)

        beta = self.make_purchase_sensitivity.weights
        x    = numpy.array((1,
                            time_since_last_transaction,
                            last_viewed_offer_strength,
                            viewed_active_offer))
        p = 1.0 / (1.0 + numpy.exp(-numpy.dot(beta, x)))

        # logging.debug('        beta = {}'.format(beta))
        # logging.debug('           x = {}'.format(x))
        # logging.debug('      beta*x = {}'.format(beta*x))
        # logging.debug('dot(beta, x) = {}'.format(numpy.dot(beta, x)))
        # logging.debug('           p = {}'.format(p))

        # flip a coin to decide if a purchase was made
        made_purchase = True if numpy.random.random() < p else False

        if made_purchase:
            # logging.debug('Made purchase')
            # Determine if this is an outlier order or regular order
            if numpy.random.random() < self.outlier_frequency:
                purchase_amount = self.outlier_purchase_amount(world)
            else:
                purchase_amount = self.purchase_amount(world)

            transaction = Transaction(world.world_time, amount=purchase_amount)
            self.history.append(transaction)
            self.last_transaction = transaction
        else:
            transaction = None

        return transaction


    def purchase_amount(self, world):
        """Randomly sample a gamma distribution with special outliers determine the amount of a purchase.

        Mean purchase amount depends on: average transaction amount, segment, income, taste, offer

        Order of components in sensitivity and value arrays (beta and x) is:
        0: average_transaction_amount (value = 1, this is the bias / constant term)
        1: income (sensitivity = 1, the relationship is nonlinear and captured by the special "income_curve" function)
        2: marketing_segment components
        3: taste components
        """

        # logging.debug('Purchase amount decision at time t = {}'.format(world.world_time))

        # average purchase increases with income, but has a minimum (min price of a product) and tops out at some level:
        # linear with min and max plateaus at thresholds
        min_income = 10000.0
        max_income = 100000.0
        min_mean_purchase = 1.0
        max_mean_purchase = 25.0

        income_adjusted_purchase_sensitivity = self.bounded_response(self.income, min_income, max_income, min_mean_purchase, max_mean_purchase)

        beta = self.purchase_amount_sensitivity.weights
        x = numpy.concatenate((numpy.array((1,
                                            income_adjusted_purchase_sensitivity)),
                               self.marketing_segment.weights,
                               self.taste.weights))

        # Cannot allow a non-positive mean purchase, so set the lower limit
        mean = max(min_mean_purchase, numpy.dot(beta, x))

        # logging.debug('        beta = {}'.format(beta))
        # logging.debug('           x = {}'.format(x))
        # logging.debug('      beta*x = {}'.format(beta*x))
        # logging.debug('dot(beta, x) = {}'.format(numpy.dot(beta, x)))
        # logging.debug('        mean = {}'.format(mean))

        # simple relationship between mean and var reflects the increased options for purchase to achieve higher mean
        var = 2.0 * mean

        # mean = k*theta
        # var = k*theta**2

        theta = var / mean
        k = mean ** 2 / var

        # minimum purchase is $0.05
        # all purchases are rounded to a whole number of cents
        amount = max(0.05, round(numpy.random.gamma(shape=k, scale=theta, size=None), 2))

        # logging.debug('Purcahse amount = {}'.format(amount))

        return amount


    def outlier_purchase_amount(self, world):
        """Randomly sample an outlier distribution to determine the amount of a purchase.

        The model here is purchasing for a group of mean size 30 (poisson)
        """

        mean_group_size = 30
        group_size = numpy.random.poisson(mean_group_size)

        # logging.debug('Purchase amount decision at time t = {}'.format(world.world_time))

        # average purchase increases with income, but has a minimum (min price of a product) and tops out at some level:
        # linear with min and max plateaus at thresholds
        min_income = 10000.0
        max_income = 100000.0
        min_mean_purchase = 1.0
        max_mean_purchase = 25.0

        income_adjusted_purchase_sensitivity = self.bounded_response(self.income, min_income, max_income, min_mean_purchase, max_mean_purchase)

        beta = self.purchase_amount_sensitivity.weights
        x = numpy.concatenate((numpy.array((1,
                                            income_adjusted_purchase_sensitivity)),
                               self.marketing_segment.weights,
                               self.taste.weights))

        # Cannot allow a non-positive mean purchase, so set the lower limit
        mean = max(min_mean_purchase, numpy.dot(beta, x))

        # logging.debug('        beta = {}'.format(beta))
        # logging.debug('           x = {}'.format(x))
        # logging.debug('      beta*x = {}'.format(beta*x))
        # logging.debug('dot(beta, x) = {}'.format(numpy.dot(beta, x)))
        # logging.debug('        mean = {}'.format(mean))

        # simple relationship between mean and var reflects the increased options for purchase to achieve higher mean
        var = 2.0 * mean

        # mean = k*theta
        # var = k*theta**2

        theta = var / mean
        k = mean ** 2 / var

        # minimum purchase is $0.05
        # all purchases are rounded to a whole number of cents
        amount = max(0.05, round(numpy.sum(numpy.random.gamma(shape=k, scale=theta, size=group_size)), 2))

        # logging.debug('Purcahse amount = {}'.format(amount))

        return amount


class TestPerson(unittest.TestCase):
    """Test class for Person."""

    def setUp(self):
        self.offer = Offer(10,
                           channel=Categorical(('web', 'email', 'mobile', 'social'), (0, 1, 1, 1)),
                           offer_type=Categorical(('bogo', 'discount', 'informational'), (0, 0, 1)))
        self.transaction = Transaction(20, amount=1.00)
        self.world = World()

        self.person = Person(became_member_on='20170101',
                             history=[self.offer, self.transaction])


    def test_person(self):
        self.assertTrue(self.person)


    def test_simulate(self):
        self.person.update(self.world)
        self.assertTrue(1)


    def test_bounded_response(self):
        min_x = 2
        max_x = 5
        f_of_min_x = 10
        f_of_max_x = 20

        x = min_x - 1
        self.assertTrue(Person.bounded_response(x, min_x, max_x, f_of_min_x, f_of_max_x) == f_of_min_x)

        x = min_x
        self.assertTrue(Person.bounded_response(x, min_x, max_x, f_of_min_x, f_of_max_x) == f_of_min_x)

        x = 0.5*(min_x + max_x)
        self.assertTrue(Person.bounded_response(x, min_x, max_x, f_of_min_x, f_of_max_x) == 0.5*(f_of_min_x + f_of_max_x))

        x = max_x
        self.assertTrue(Person.bounded_response(x, min_x, max_x, f_of_min_x, f_of_max_x) == f_of_max_x)

        x = max_x + 1
        self.assertTrue(Person.bounded_response(x, min_x, max_x, f_of_min_x, f_of_max_x) == f_of_max_x)


    def test_view_offer(self):
        person_view_offer_sensitivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'], [0, -1, 1, 1, 1, 1])
        offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
        offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 1, 0))

        discount = Offer(0, valid_from=10, valid_until=20, difficulty=10, reward=2, channel=offer_channel, offer_type=offer_type)
        person = Person(became_member_on='20170716', view_offer_sensitivity=person_view_offer_sensitivity)
        person.last_unviewed_offer = discount

        world = copy.deepcopy(self.world)
        world.world_time = 0

        person.view_offer(world)

        self.assertTrue(True)


    def test_serializaton(self):
        person_dict = self.person.to_serializable()
        person_reconstituted = Person.from_dict(person_dict)
        person_reconstituted_dict = person_reconstituted.to_serializable()
        self.assertTrue(person_reconstituted_dict == person_dict)

        person_json = self.person.to_json()
        person_reconstituted = Person.from_json(person_json)
        person_reconstituted_dict = person_reconstituted.to_serializable()
        self.assertTrue(person_reconstituted_dict == person_dict)
