import unittest

class Employee:
    def __init__(self, first_name, last_name, earning):
        self.first_name = first_name
        self.last_name = last_name
        self.earning = earning

    def give_rise(self, rise=5000):
        self.earning += rise

class TextEarningRise(unittest.TestCase):

    def setUp(self) -> None:

        self.employee = Employee('HAL','kiki',10000)
        self.expect_earning = 13000

    def test_give_raise(self):

        self.employee.give_rise(3000)
        self.assertEqual(self.employee.earning, self.expect_earning)

if __name__ == '__main__':
    unittest.main()
