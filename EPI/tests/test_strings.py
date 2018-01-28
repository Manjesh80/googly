from ..strings_py import *


# 6.7 Find all mnemonics of a phone number
# pytest EPI\tests\test_strings.py::test_phone_combination
def test_phone_combination():
    res = phone_combination('23')
    assert len(res) == 9


# 6.9 Convert from Roman to Decimal
# pytest EPI\tests\test_strings.py::test_phone_combination
def test_roman_to_decimal():
    pass


# 6.10 Compute all valid IP address
# pytest EPI\tests\test_strings.py::test_get_valid_ip_addresses
def test_get_valid_ip_addresses():
    res = get_valid_ip_addresses_rec('19216811', 4)
    assert len(res) == 9


# 6.11 Compute snake string
# pytest EPI\tests\test_strings.py::test_snake_string
def test_snake_string():
    assert snake_string("Hello World") == 'e lHloWrdlo'


# 6.13 Match sub string rolling
# pytest EPI\tests\test_strings.py::test_sub_string_index
def test_sub_string_index():
    assert sub_string_index('AABCD', 'BCD') == 2
