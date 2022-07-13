import unittest
from city_functions import city_info

class CityTestCase(unittest.TestCase):

    def test_city_info(self):
        city = city_info('Santiago', 'Chile')
        self.assertEqual(city, 'Santiago,Chile - population 500000')

if __name__ == '__main__':
    unittest.main()