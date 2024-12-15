import textwrap
from test_file import TestFile
from colorama import Fore, Back, Style, init
init(autoreset=True)


class Function:
    def __init__(self, function_name, function_definition):
        self.function_name = function_name
        self.function_definition = function_definition
        self.function_test_files = []

    def review_function(self):
        print(Fore.LIGHTGREEN_EX + f'Function Name:\n\t{self.function_name}')
        choice = input(Fore.WHITE + 'Exit: e | Exit and Save: s | Continue: enter key\n' + Fore.LIGHTBLUE_EX
                       + '\tChoose: ')
        if choice == 'e':
            return True
        if choice == 's':
            exit_tuple = (True, self.function_name)
            return exit_tuple
        else:
            print(Fore.LIGHTGREEN_EX + f'Function Definition:\n\t' + '\n\t'.join(self.function_definition) + '\n\n')
        exit_tuple = (False, self.function_name)
        return exit_tuple

    def quiz_function(self):
        print(Fore.LIGHTGREEN_EX + f'Function Name:\n\t{self.function_name}')
        choice = input(Fore.WHITE + 'Exit: e | Exit and Save: s | Continue: enter key\n' + Fore.LIGHTBLUE_EX
                       + '\tChoose: ')
        if choice == 'e':
            return True
        if choice == 's':
            exit_tuple = (True, self.function_name)
            return exit_tuple
        else:
            go = True
            while go:
                user_answer = input(Fore.WHITE + 'Please enter the definition of the function:\n\n' + Fore.LIGHTBLUE_EX)
                if user_answer.lower() == ' '.join(self.function_definition).lower():
                    print(Fore.GREEN + '\nCorrect Answer\n')
                    go = False
                else:
                    print(Fore.RED + '\nWrong Answer\n')
                if go:
                    print(Fore.LIGHTGREEN_EX + f'Function Definition:\n\t' + '\n\t'.join(self.function_definition) +
                          '\n\n')
                    book_string_set = set(' '.join(self.function_definition).split())
                    user_string = set(user_answer.split())
                    str_diff = book_string_set.symmetric_difference(user_string)
                    print(Fore.RED + f'Differences between book string and user string:\n\t{str_diff}')
        exit_tuple = (False, self.function_name)
        return exit_tuple

    def add_test_files_to_func(self, resume=None):
        print(Fore.LIGHTGREEN_EX + f'Function Name:\n\t{self.function_name}')
        choice = input(Fore.WHITE + 'Exit: e | Exit and Save: s | Continue Adding Test Files to Functions: enter key\n'
                       + Fore.LIGHTBLUE_EX + '\tChoose: ')
        if choice == 'e':
            return True
        if choice == 's':
            exit_tuple = (True, self.function_name)
            return exit_tuple
        else:
            print(Fore.LIGHTGREEN_EX + f'Function Definition:\n\t' + '\n\t'.join(self.function_definition) + '\n\n')
        if not self.function_test_files:
            self.func_create_test_file()
        exit_tuple = (False, self.function_name)
        return exit_tuple

    def load_func_test_file_list(self, list_of_test_files):
        for ind_test_file in list_of_test_files:
            test_file_info = TestFile(ind_test_file['test_file_name'], ind_test_file['test_file_caption'])
            self.function_test_files.append(test_file_info)
            
    def func_create_test_file(self):
        name_of_test_file = input(Fore.GREEN + '\t\tWhat is the name of the test file you wish to create?\n\t\t')
        caption_of_test_file = textwrap.wrap(
            input(Fore.GREEN + f'\t\tWhat is the caption of {name_of_test_file}?\n\t\t'), 80)
        self.function_test_files.append(TestFile(name_of_test_file, caption_of_test_file))

    def class_remove_test_file(self):
        name_of_test_file = input(Fore.RED + '\t\tWhat is the name of the test file you wish to remove?\n\t\t')
        del self.function_test_files[[x.test_file_name for x in self.function_test_files].index(name_of_test_file)]

    def write_function(self):
        function_dict = {
            'function_name':
                self.function_name,
            'function_definition':
                self.function_definition,
            'function_test_files':
                [i.write_test_file() for i in self.function_test_files]
        }
        return function_dict

    def function_internal_menu(self):
        go = True
        while go:
            choice = input(Fore.YELLOW + f'Function Name: {self.function_name}\n\n' +
                           Fore.WHITE +
                           'Create A Test File: c | Remove A Test File: r | Exit: e\n\n' +
                           Fore.LIGHTBLUE_EX + '\tChoose: ')
            if choice == 'c':
                self.func_create_test_file()
            elif choice == 'r':
                self.class_remove_test_file()
            elif choice == 'e':
                go = False