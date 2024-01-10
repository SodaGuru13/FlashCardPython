import keyword
from colorama import Fore, Back, Style, init
init(autoreset=True)


def describe_bool():
    print(Fore.YELLOW + 'This is a module from Real Python describing the bool class.\n' + Fore.LIGHTMAGENTA_EX
          + '>>True + True + False + True')
    check = True + True + False + True
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>issubclass(bool, int)')
    check = issubclass(bool, int)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>isinstance(True, bool)')
    check = isinstance(True, bool)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>isinstance(False, bool)')
    check = isinstance(False, bool)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>isinstance(1, bool)')
    check = isinstance(1, bool)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>isinstance(0, bool)')
    check = isinstance(0, bool)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>import keyword\nkeyword.kwlist')
    print(keyword.kwlist)
    print(Fore.LIGHTMAGENTA_EX + '>>int(True)')
    check = int(True)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>int(False)')
    check = int(False)
    print(check)
    print(Fore.LIGHTMAGENTA_EX + '>>codes = [264, 118, 543, 705, 1152, 674]\n>>N = 3')
    codes = [264, 118, 543, 705, 1152, 674]
    N = 3
    print(Fore.LIGHTMAGENTA_EX + '>>[c % N == 0 for c in codes]')
    codes_true_false_list = [c % N == 0 for c in codes]
    print(codes_true_false_list)
    print(Fore.LIGHTMAGENTA_EX + '>>sum([c % N == 0 for c in codes])')
    check = sum([c % N == 0 for c in codes])
    print(check)


def describe_bytearray():
    print(Fore.YELLOW + 'Bytearray Class Test\n' + Fore.LIGHTMAGENTA_EX + '>>bytearray()')
    byte_array = bytearray()
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '>>bytearray(10)')
    byte_array = bytearray(10)
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '>>bytearray(range(20))')
    byte_array = bytearray(range(20))
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '>>bytearray(b\'Hi!\')')
    byte_array = bytearray(b'Hi!')
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '>>string = "Python is interesting"\n>># string with encoding \'utf-8\'\n'
          + '>>arr = bytearray(string, \'utf-8\')')
    string = "Python is interesting"
    # string with encoding 'utf-8'
    byte_array = bytearray(string, 'utf-8')
    print(byte_array)
    print(Fore.LIGHTMAGENTA_EX + '# Explaining what a iterable looks like as a bytearray.\n>>nums = [1, 2, 3, 4, 5]')
    nums = [1, 2, 3, 4, 5]
    byte_array = bytearray(nums)
    print(byte_array)


def describe_bytes():
    print(Fore.YELLOW + 'Bytes Class Test\n' +  Fore.LIGHTMAGENTA_EX
          + '>>single_quotes = b\'still allows embedded "double" quotes\'\n'
          + '>>double_quotes = b"still allows embeded \'single\' quotes"\n'
          + '>>triple_single_quoted = b\'\'\'3 single quotes\'\'\'\n' + '>>triple_double_quoted = """3 double quotes"""\n'
          + '>>zero_filled = bytes(10)\n' + '>>iterable_integers = bytes(range(20))\n'
          + '>>object_example = bytes(memoryview(iterable_integers))' + '>>string = \'Papa Mónty\''
          + '>>arr = bytes(string, \'utf-8\', errors = \'replace\')')
    single_quotes = b'still allows embedded "double" quotes'
    double_quotes = b"still allows embeded 'single' quotes"
    triple_single_quoted = b'''3 single quotes'''
    triple_double_quoted = b"""3 double quotes"""
    string = 'Papa Mónty'
    arr = bytes(string, 'ascii', errors = 'replace')
    zero_filled = bytes(10)
    iterable_integers = bytes(range(20))
    object_example = bytes(memoryview(iterable_integers))
    print(single_quotes)
    print(double_quotes)
    print(triple_single_quoted)
    print(triple_double_quoted)
    print(zero_filled)
    print(iterable_integers)
    print(object_example)
    print(arr)


def describe_complex():
    print(Fore.YELLOW + 'Complex Class Test\n' + Fore.LIGHTMAGENTA_EX + '>>comp = complex(5, 9)')
    comp = complex(5, 9)
    print(comp)
    print(Fore.LIGHTMAGENTA_EX + '>>comp = complex(\'4 + 7j\')')
    comp = complex('4+7j')
    print(comp)
    print(Fore.LIGHTMAGENTA_EX + '>>comp = complex(4 + 5j, 5 + 4j)')
    comp = complex(4+5j, 5+4j)
    print(comp)
    print(Fore.LIGHTMAGENTA_EX + '>>comp = complex(8)')
    comp = complex(8)
    print(comp)
    print(Fore.LIGHTMAGENTA_EX + '>>comp = complex()')
    comp = complex()
    print(comp)


def main():
    describe_bool()
    describe_bytearray()
    describe_bytes()
    describe_complex()


if __name__ == '__main__':
    main()