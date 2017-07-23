"""A Collection of handy functions for working with decision simulations that don't seem to fit anywhere else."""
import unittest

import os
import copy
import numpy


def mkdir_if_missing(path):
    """Creates directory tree ending in path if it doesn't already exist."""
    if not os.path.isdir(path):
        if os.path.exists(path):
            raise ValueError('ERROR - the path name {} is already in use by a file or link.'.format(path))
        else:
            os.makedirs(path)


class ProfileGenerator(object):
    """Generate profile data for initializing populations."""

    def __init__(self):
        self.initialize_age_gender_joint_distribution()


    def initialize_age_gender_joint_distribution(self):
        self.age_gender_joint_distribution = self.get_us_census_2017_get_age_gender_joint_distribution()


    def get_us_census_2017_get_age_gender_joint_distribution(self):
        data_file = 'US_Census_2017_projected_age_sex_totals.csv'
        delimiter = ','
        required_header = ['age', 'M', 'F']
        with open(data_file, 'r') as input_file:
            joint_distribution = list()
            for line_number, line in enumerate(input_file):
                data = map(lambda s: s.strip(), line.split(delimiter))
                if line_number == 0:
                    assert data == required_header, 'ERROR - data file has wrong header: {}'.format(data)
                else:
                    age, n_M, n_F = map(int, data)
                    joint_distribution.append(([age, 'M'], n_M))
                    joint_distribution.append(([age, 'F'], n_F))

        return joint_distribution


    # def sample_optin_age_gender(self,
    #                             optin_fraction=1.00,
    #                             age_range=None, default_age=None,
    #                             fixed_gender=None, default_gender=None):




    def sample_age_gender(self,
                          size,
                          non_binary_fraction=0.00,
                          min_age=None, max_age=None,
                          fixed_gender=None):
        return self.sample_us_census_2017_age_gender_joint_distribution(size, non_binary_fraction, min_age, max_age, fixed_gender)


    def sample_us_census_2017_age_gender_joint_distribution(self,
                                                            size,
                                                            non_binary_fraction=0.00,
                                                            min_age=None, max_age=None,
                                                            fixed_gender=None):
        """Draws a random (age, gender) pair based on U.S. Census projections for 2017.

        The sameple can be restricted to a range of ages and a single gender if desired.
        """
        classes, weights = zip(*self.age_gender_joint_distribution)
        # loop through classes and discard any that are not in the specified ranges.
        if (min_age is not None) or (max_age is not None) or (fixed_gender is not None):
            if min_age is None:
                min_age = -1
            if max_age is None:
                max_age = 101
            if fixed_gender is None:
                genders = ('M', 'F')
            elif fixed_gender not in ('M', 'F'):
                print 'WARNING - You can only specify M or F as the fixed gender, not {}. Using (M, F)'.format(fixed_gender)
                genders = ('M', 'F')
            else:
                genders = (fixed_gender,)

            # drop the record for classes that are out of range
            restricted_classes = list()
            restricted_weights = list()
            for cls, weight in zip(classes, weights):
                age, gender = cls
                if (min_age <= age <= max_age) and (gender in genders):
                    restricted_classes.append(cls)
                    restricted_weights.append(weight)
        else:
            restricted_classes = classes
            restricted_weights = weights

        # compute population fractions
        Z = float(sum(restricted_weights))
        p = [wi/Z for wi in restricted_weights]

        # sample
        if size > 1:
            idx = numpy.random.choice(range(len(restricted_classes)), size=size, p=p)
            sample = [restricted_classes[id] for id in idx]
        else:
            idx = numpy.random.choice(range(len(restricted_classes)), p=p)
            sample = restricted_classes[idx]

        # randomly select and modify gender
        if 0.0 <= non_binary_fraction <= 1.0:
            if size > 1:
                for i in range(len(sample)):
                    if numpy.random.random() < non_binary_fraction:
                        # change gender to 'O', be careful to work on copies of these objects instead of references
                        sample[i] = copy.deepcopy(sample[i])
                        sample[i][1] = 'O'
                        sample[i] = tuple(sample[i])
            else:
                if numpy.random.random() < non_binary_fraction:
                    # change gender to 'O', be careful to work on copies of these objects instead of references
                    sample = copy.deepcopy(sample)
                    sample[1] = 'O'
                    sample = tuple(sample)

        else:
            raise ValueError('ERROR - non_binary_fraction = {}, which does not lies in the interval [0, 1]'.format(non_binary_fraction))

        return sample


class TestUtilities(unittest.TestCase):
    """Test class for utilities.py"""

    def setUp(self):
        self.pg = ProfileGenerator()


    def test_sample_age_gender(self):
        from collections import Counter

        sample = self.pg.sample_age_gender(10000, non_binary_fraction=0.10)

        ctr = Counter(sample)
        n = sum(ctr.values())
        n_M = sum([v for k, v in ctr.iteritems() if k[1] == 'M'])
        n_F = sum([v for k, v in ctr.iteritems() if k[1] == 'F'])
        n_O = sum([v for k, v in ctr.iteritems() if k[1] == 'O'])
        ages = reduce(lambda a, b: a + b, [[k[0],]*v for k, v in ctr.iteritems()])
        print 'total population: {}'.format(n)
        print 'fraction M/F/O: {:6.4}/{:6.4}/{:6.4}'.format(n_M/float(n), n_F/float(n), n_O/float(n))
        print 'Age stats:'
        print '    min:    {:5}'.format(numpy.min(ages))
        print '    mean:   {:5.4}'.format(numpy.mean(ages))
        print '    median: {:5}'.format(numpy.median(ages))
        print '    max:    {:5}'.format(numpy.max(ages))
        print
        print 'Age deciles:'
        pct = range(0, 101, 10)
        for ptile, val in zip(pct, numpy.percentile(ages, pct)):
            print '      {:3}: {:4}'.format(ptile, val)

        self.assertTrue(True)

    def test_restricted_sample_age_gender(self):
        from collections import Counter

        sample = self.pg.sample_age_gender(10000,
                                           non_binary_fraction=0.10,
                                           min_age=25, max_age=35,
                                           fixed_gender='F')

        ctr = Counter(sample)
        n = sum(ctr.values())
        n_M = sum([v for k, v in ctr.iteritems() if k[1] == 'M'])
        n_F = sum([v for k, v in ctr.iteritems() if k[1] == 'F'])
        n_O = sum([v for k, v in ctr.iteritems() if k[1] == 'O'])
        ages = reduce(lambda a, b: a + b, [[k[0], ] * v for k, v in ctr.iteritems()])
        print 'total population: {}'.format(n)
        print 'fraction M/F/O: {:6.4}/{:6.4}/{:6.4}'.format(n_M / float(n), n_F / float(n), n_O / float(n))
        print 'Age stats:'
        print '    min:    {:5}'.format(numpy.min(ages))
        print '    mean:   {:5.4}'.format(numpy.mean(ages))
        print '    median: {:5}'.format(numpy.median(ages))
        print '    max:    {:5}'.format(numpy.max(ages))
        print
        print 'Age deciles:'
        pct = range(0, 101, 10)
        for ptile, val in zip(pct, numpy.percentile(ages, pct)):
            print '      {:3}: {:4}'.format(ptile, val)

        self.assertTrue(True)
