"""Run an experiment in which a portfolio of offers is deployed to a population."""

import logging
import unittest

import argparse
import os.path
import glob
import datetime
import numpy
import shutil

from externalities import World, Offer, Transaction, Event, Categorical
from utilities import ProfileGenerator, mkdir_if_missing
from person import Person
from population import Population

dt_fmt = '%Y%m%d'
now = datetime.datetime.now()


def g(x):
    return 1.0 / (1.0 + numpy.exp(-x))

def g_inv(y):
    return numpy.log(y / (1.0 - y))


def create_people(n, parameters):

    known_parameters = {'profile_optout_rate',
                        'min_tenure',
                        'max_tenure',
                        'min_age',
                        'max_age',
                        'f_fraction',
                        'non_binary_fraction',
                        'min_income',
                        'max_income',
                        'view_offer_sensitivity',
                        'make_purchase_sensitivity',
                        'purchase_amount_sensitivity'}

    if not set(parameters.keys()).issubset(known_parameters):
        print 'WARNING - Unrecognized parameters in create people parameters: {}'.format(known_parameters - set(parameters.keys()))

    default_view_offer_sensitivity = Categorical(
        ['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
        [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 1, 1, 1, 1])

    default_make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 24.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 1, 1])

    default_purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    profile_optout_rate = parameters.get('profile_optout_rate', 0.3)
    min_tenure = parameters.get('min_tenure', 0)
    max_tenure = parameters.get('max_tenure', 5*365)
    min_age = parameters.get('min_age', 18)
    max_age = parameters.get('max_age', None)
    f_fraction = parameters.get('f_fraction', None)
    non_binary_fraction = parameters.get('non_binary_fraction', 0.03)
    min_income = parameters.get('min_income',  10000)
    max_income = parameters.get('max_income', 300000)
    view_offer_sensitivity = parameters.get('view_offer_sensitivity', default_view_offer_sensitivity)
    make_purchase_sensitivity = parameters.get('make_purchase_sensitivity', default_make_purchase_sensitivity)
    purchase_amount_sensitivity = parameters.get('purchase_amount_sensitivity', default_purchase_amount_sensitivity)

    # fixed income parameters set to resemble US household income distribution
    income_threshold = 10000
    beta = 1.0 / 0.0004

    pg = ProfileGenerator()

    people = list()
    for i in range(n):
        became_member_on = (now - datetime.timedelta(days=numpy.random.choice(range(min_tenure, max_tenure)))).strftime(dt_fmt)

        if numpy.random.random() < 1.0 - profile_optout_rate:
            if f_fraction is not None:
                if numpy.random.random() < f_fraction:
                    age, gender = pg.sample_age_gender(1, non_binary_fraction, min_age, max_age, fixed_gender='F')
                else:
                    age, gender = pg.sample_age_gender(1, non_binary_fraction, min_age, max_age, fixed_gender='M')
            else:
                    age, gender = pg.sample_age_gender(1, non_binary_fraction, min_age, max_age)
            dob = (now - datetime.timedelta(days=int(age*365.25 + numpy.random.choice(range(365))))).strftime(dt_fmt)
            income = None
            for i in range(25):
                x = max(income_threshold, numpy.random.exponential(beta))
                if min_income <= x <= max_income:
                    income = x
                    break
        else:
            dob = None
            gender = None
            income = None

        people.append(Person(became_member_on,
                             dob=dob,
                             gender=gender,
                             income=income,
                             view_offer_sensitivity=view_offer_sensitivity,
                             make_purchase_sensitivity=make_purchase_sensitivity,
                             purchase_amount_sensitivity=purchase_amount_sensitivity))

    return people


def create_people_0(n):
    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 0, 1, 1, 2])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 24.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 1, 1])

    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.1,
                  'min_tenure': 180,
                  'max_tenure': 3*365,
                  'min_age': 18,
                  'max_age': None,
                  'f_fraction': None,
                  'non_binary_fraction': 0.02,
                  'min_income': 30000,
                  'max_income': 75000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_people_1(n):
    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.50) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 1, 1, 1, 1])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 48.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 1, 1])

    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.2,
                  'min_tenure': 180,
                  'max_tenure': 3*365,
                  'min_age': 36,
                  'max_age': None,
                  'f_fraction': None,
                  'non_binary_fraction': 0.04,
                  'min_income':  50000,
                  'max_income': 100000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_people_2(n):

    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 1, 0, 1, 2])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 48.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 1, 1])

    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.2,
                  'min_tenure': 180,
                  'max_tenure': 3*365,
                  'min_age': 48,
                  'max_age': None,
                  'f_fraction': 0.67,
                  'non_binary_fraction': 0.00,
                  'min_income':  70000,
                  'max_income': 120000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_people_3(n):

    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.20) - 2, -abs(g_inv(0.01) / float(1 * 24 * 30)), 0, 0.5, 0.5, 1])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 24.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 0.5, 0.5])

    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.1,
                  'min_tenure': 0,
                  'max_tenure': 1*365,
                  'min_age': 18,
                  'max_age': None,
                  'f_fraction': None,
                  'non_binary_fraction': 0.02,
                  'min_income':  30000,
                  'max_income':  75000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_people_4(n):
    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.50) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 1, 1, 1, 1])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 48.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 0.1, 0.1])

    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.2,
                  'min_tenure': 0,
                  'max_tenure': 1*365,
                  'min_age': 36,
                  'max_age': None,
                  'f_fraction': None,
                  'non_binary_fraction': 0.04,
                  'min_income':  50000,
                  'max_income': 100000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_people_5(n):
    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 1, 0, 1, 2])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 48.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 0, 0])

    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.2,
                  'min_tenure': 0,
                  'max_tenure': 1*365,
                  'min_age': 48,
                  'max_age': None,
                  'f_fraction': 0.67,
                  'non_binary_fraction': 0.00,
                  'min_income':  70000,
                  'max_income': 120000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_people_6(n):
    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 0, 0, 1, 3])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 48.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 0, 1])
    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.01,
                  'min_tenure': 0,
                  'max_tenure': 1*365,
                  'min_age': 18,
                  'max_age': None,
                  'f_fraction': 0.20,
                  'non_binary_fraction': 0.01,
                  'min_income': 30000,
                  'max_income': 75000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_people_7(n):
    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 0, 0, 2, 2])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 24.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 0.5, 1.5])
    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.01,
                  'min_tenure': 180,
                  'max_tenure': 5*365,
                  'min_age': 18,
                  'max_age': None,
                  'f_fraction': 0.20,
                  'non_binary_fraction': 0.01,
                  'min_income': 30000,
                  'max_income': 75000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_people_8(n):
    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 1, 2, 1, 0])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 36.0), abs(g_inv(0.10) / float(1 * 24 * 30)), -1, -1])
    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.3,
                  'min_tenure': 0,
                  'max_tenure': 1*365,
                  'min_age': 40,
                  'max_age': None,
                  'f_fraction': 0.33,
                  'non_binary_fraction': 0.02,
                  'min_income':  50000,
                  'max_income': 100000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_people_9(n):
    view_offer_sensistivity = Categorical(['background', 'offer_age', 'web', 'email', 'mobile', 'social'],
                [g_inv(0.20) - 4, -abs(g_inv(0.01) / float(1 * 24 * 30)), 1, 2, 1, 0])
    make_purchase_sensitivity = Categorical(
        ['background', 'time_since_last_transaction', 'last_viewed_offer_strength', 'viewed_active_offer'],
        [g_inv(1.0 / 36.0), abs(g_inv(0.10) / float(1 * 24 * 30)), 0.5, 0.5])
    purchase_amount_sensitivity = Categorical(
        ['background', 'income_adjusted_purchase_sensitivity', 'front page', 'local', 'entertainment', 'sports',
         'opinion', 'comics', 'sweet', 'sour', 'salty', 'bitter', 'umami'], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    parameters = {'profile_optout_rate': 0.1,
                  'min_tenure': 270,
                  'max_tenure': 5*365,
                  'min_age': 40,
                  'max_age': None,
                  'f_fraction': 0.33,
                  'non_binary_fraction': 0.02,
                  'min_income':  50000,
                  'max_income': 100000,
                  'view_offer_sensitivity': view_offer_sensistivity,
                  'make_purchase_sensitivity': make_purchase_sensitivity,
                  'purchase_amount_sensitivity': purchase_amount_sensitivity}

    return create_people(n, parameters)


def create_portfolio():

    # Month 1 offers
    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 0))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (1, 0, 0))
    bogo_a = Offer(0, valid_from=0, valid_until=1*7*24, difficulty=5, reward=5, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (1, 0, 0))
    bogo_b = Offer(0, valid_from=0, valid_until=1*7*24, difficulty=10, reward=10, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 0))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 1, 0))
    discount_c = Offer(0, valid_from=0, valid_until=1*7*24, difficulty=10, reward=2, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 1, 0))
    discount_d = Offer(0, valid_from=0, valid_until=1*7*24, difficulty=7, reward=3, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 0, 1))
    # valid for 1 day = 1*24 hours
    info_e = Offer(0, valid_from=0, valid_until=3*24, difficulty=0, reward=0, channel=offer_channel, offer_type=offer_type)

    # Month 2 offers
    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (1, 0, 0))
    bogo_z = Offer(0, valid_from=0, valid_until=5*24, difficulty=5, reward=5, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (1, 0, 0))
    bogo_y = Offer(0, valid_from=0, valid_until=5*24, difficulty=10, reward=10, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 0))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 1, 0))
    discount_x = Offer(0, valid_from=0, valid_until=10*24, difficulty=20, reward=5, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 1))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 1, 0))
    discount_w = Offer(0, valid_from=0, valid_until=10*24, difficulty=10, reward=2, channel=offer_channel, offer_type=offer_type)

    offer_channel = Categorical(('web', 'email', 'mobile', 'social'), (1, 1, 1, 0))
    offer_type = Categorical(('bogo', 'discount', 'informational'), (0, 0, 1))
    info_v = Offer(0, valid_from=0, valid_until=4*24, difficulty=0, reward=0, channel=offer_channel, offer_type=offer_type)

    portfolio = {'historical - bogo_a': bogo_a,
                 'historical - bogo_b': bogo_b,
                 'historical - discount_c': discount_c,
                 'historical - discount_d': discount_d,
                 'historical - info_e': info_e,
                 'new - bogo_z': bogo_z,
                 'new - bogo_y': bogo_y,
                 'new - discount_x': discount_x,
                 'new - discount_w': discount_w,
                 'new - info_v': info_v}

    return portfolio


def assign_offers_to_subpopulation(population, subpopulation, deliveries_file_name, deliveries_log_file_name, control_fraction=0.25, delimiter='|', clean_path=True):

    if clean_path:
        data_file_names = glob.glob(os.path.join(population.deliveries_path, '*'))
        for data_file_name in data_file_names:
            os.remove(data_file_name)

    offer_ids = population.portfolio.keys()

    # update the validity dates of the offers
    now = population.world.world_time
    for offer in population.portfolio.values():
        offer_length = offer.valid_until - offer.valid_from
        offer.valid_from = now
        offer.valid_until = now + offer_length

    # make random delivery decisions
    deliveries = list()
    for person_id in subpopulation:
        assert person_id in population.people, 'ERROR - an indiviual in the desired subpopulation is not part of the general population: {}'.format(person_id)
        # hold out a fraction of people as control
        if numpy.random.random() < 1.0 - control_fraction:
            # make random deliveries to the rest
            deliveries.append((person_id, numpy.random.choice(offer_ids)))

    # write the deliveries file
    with open(deliveries_file_name, 'w') as deliveries_file:
        for delivery in deliveries:
            print >> deliveries_file, delimiter.join(map(str, delivery))

    # make a copy of the delivery file
    shutil.copy(deliveries_file_name, deliveries_log_file_name)


def assign_oracle_offers_to_subpopulation(population, subpopulation, optimal_offer_id, deliveries_file_name, deliveries_log_file_name, control_fraction=0.25, delimiter='|', clean_path=True):

    if clean_path:
        data_file_names = glob.glob(os.path.join(population.deliveries_path, '*'))
        for data_file_name in data_file_names:
            os.remove(data_file_name)

    offer_ids = population.portfolio.keys()

    # update the validity dates of the offers
    now = population.world.world_time
    for offer in population.portfolio.values():
        offer_length = offer.valid_until - offer.valid_from
        offer.valid_from = now
        offer.valid_until = now + offer_length

    # make random delivery decisions
    deliveries = list()
    for person_id in subpopulation:
        assert person_id in population.people, 'ERROR - an indiviual in the desired subpopulation is not part of the general population: {}'.format(person_id)
        # hold out a fraction of people as control
        if numpy.random.random() < 1.0 - control_fraction:
            # make random deliveries to the rest
            deliveries.append((person_id, optimal_offer_id))

    # write the deliveries file
    with open(deliveries_file_name, 'w') as deliveries_file:
        for delivery in deliveries:
            print >> deliveries_file, delimiter.join(map(str, delivery))

    # make a copy of the delivery file
    shutil.copy(deliveries_file_name, deliveries_log_file_name)


def main(args):
    world = World(real_time_tick=0.000, world_time_tick=6)

    people = list()
    people.extend(create_people_0(1000))
    people.extend(create_people_1(1000))
    people.extend(create_people_2(1000))
    people.extend(create_people_3(1000))
    people.extend(create_people_4(1000))
    people.extend(create_people_5(1000))
    people.extend(create_people_6(1000))
    people.extend(create_people_7(1000))
    people.extend(create_people_8(1000))
    people.extend(create_people_9(1000))

    portfolio = create_portfolio()

    delivery_log_path = os.path.join(args.data_path, 'delivery_log')
    deliveries_path = os.path.join(args.data_path, 'delivery')
    transcripts_file_name = os.path.join(args.data_path, 'transcript.json')
    population_file_name = os.path.join(args.data_path, 'population.json')

    # clean up from previous runs if there's data present
    old_files = glob.glob(os.path.join(args.data_path, '*'))

    for file_name in old_files:
        if os.path.isfile(file_name):
            os.remove(file_name)

    # create directories if they don't already exist
    mkdir_if_missing(args.data_path)
    mkdir_if_missing(delivery_log_path)
    mkdir_if_missing(deliveries_path)

    # initialize
    population = Population(world,
                            people=people,
                            portfolio=portfolio.values(),
                            deliveries_path=deliveries_path,
                            transcript_file_name=transcripts_file_name)

    with open(population_file_name, 'w') as population_file:
        print >> population_file, population.to_json()

    for offer_name, offer in sorted(portfolio.iteritems()):
        print '{}: {}'.format(offer_name, offer.id)
    print

    # find the id for the optimal offer
    optimal_offer_id = None
    for offer_id, offer in population.portfolio.iteritems():
        if (offer.offer_type.get('bogo') == 1):
            if all([c == 1 for c in offer.channel.weights]):
                optimal_offer_id = offer_id
                break
    assert optimal_offer_id is not None, 'ERROR - failed to find the optimal offer'

    # main simulatio loop
    has_received_offer = set()
    subpop_remaining = set(population.people.keys()) - has_received_offer

    sample_size_per_day = int(len(population.people) / 6.0)

    for day in range(30):
        print 'Starting day {}, subpop_len={}, sample_size_per_day={}'.format(day, len(subpop_remaining), sample_size_per_day)

        # random deliveries each day for 6 days
        if len(subpop_remaining) > 0:
            deliveries_file_short_name = 'deliveries.day_{}.csv'.format(day)
            deliveries_file_name = os.path.join(deliveries_path, deliveries_file_short_name)
            deliveries_log_file_name = os.path.join(delivery_log_path, deliveries_file_short_name)

            subpop = numpy.random.choice(list(subpop_remaining), size=min(sample_size_per_day, len(subpop_remaining)), replace=False)

            # make deliveries
            assign_offers_to_subpopulation(population, subpop, deliveries_file_name, deliveries_log_file_name)
            # assign_oracle_offers_to_subpopulation(population, subpop, optimal_offer_id, deliveries_file_name, deliveries_log_file_name)


            has_received_offer = has_received_offer.union(set(subpop))
            subpop_remaining = set(population.people.keys()) - has_received_offer

        # update world by one day (4 ticks per day * 1 day)
        population.simulate(n_ticks=4*1, n_proc=args.n_proc)

    return 0


def get_args():
    """Build arg parser and get command line arguments

    :return: parsed args namespace
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="data",           help="data path file name")
    parser.add_argument("--n-proc",    default=1,      type=int, help="number of Processes to use for simulation")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(get_args())

