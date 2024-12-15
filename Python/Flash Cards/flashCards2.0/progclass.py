import textwrap
from function import Function
from test_file import TestFile
from colorama import Fore, Back, Style, init
init(autoreset=True)


class ProgramClass:
    def __init__(self, class_name, class_definition):
        self.class_name = class_name
        self.class_definition = class_definition
        self.class_functions = []
        self.class_test_files = []

    def review_class(self, resume=None):
        print(Fore.LIGHTMAGENTA_EX + f'Class Name:\n\t{self.class_name}')
        choice = input(Fore.WHITE + 'Exit: e | Exit and Save: s | Continue Reviewing Classes: enter key\n'
                       + Fore.LIGHTBLUE_EX + '\tChoose: ')
        if choice == 'e':
            return True
        if choice == 's':
            class_tuple = (True, self.class_name)
            return class_tuple
        else:
            if resume is not None and 'function_name' in resume:
                start = [x.function_name for x in self.class_functions].index(resume['function_name'])
            else:
                start = 0
            print(Fore.LIGHTMAGENTA_EX + f'Class Definition:\n\t' + '\n\t'.join(self.class_definition) + '\n\n')
            end_function = None
            hit_break = False
            for info_function in self.class_functions[start:]:
                end_function = info_function.review_function()
                if type(end_function) != tuple and end_function is True:
                    hit_break = True
                    break
                elif type(end_function) == tuple and end_function[0] is True:
                    end_function += (self.class_name, )
                    return end_function
        if hit_break is True:
            return False
        if end_function is None:
            return False
        elif end_function[len(end_function)-1] == self.class_functions[len(self.class_functions)-1].function_name:
            end_function += (self.class_name, )
            return end_function
        else:
            exit_tuple = (False, self.class_name)
            return exit_tuple

    def quiz_class(self, resume=None):
        print(Fore.MAGENTA + f'Class Name:\n\t{self.class_name}')
        choice = input(Fore.WHITE + 'Exit: e | Exit and Save: s | Continue Quiz Over Classes: enter key\n' + Fore.LIGHTBLUE_EX +
                       '\tChoose: ')
        if choice == 'e':
            return True
        if choice == 's':
            class_tuple = (True, self.class_name)
            return class_tuple
        else:
            go = True
            while go:
                user_answer = input(Fore.WHITE + 'Please enter the definition of the class:\n\n' + Fore.LIGHTBLUE_EX)
                if user_answer.lower() == ' '.join(self.class_definition).lower():
                    print(Fore.GREEN + '\nCorrect Answer\n')
                    go = False
                else:
                    print(Fore.RED + '\nWrong Answer\n')
                if go:
                    print(Fore.LIGHTGREEN_EX + f'Class Definition\n\t' + '\n\t'.join(self.class_definition) + '\n\n')
                    book_string_set = set(' '.join(self.class_definition).split())
                    user_string = set(user_answer.split())
                    str_diff = book_string_set.symmetric_difference(user_string)
                    print(Fore.RED + f'Differences between book string and user string:\n\t{str_diff}')
            if resume is not None and 'function_name' in resume:
                start = [x.function_name for x in self.class_functions].index(resume['function_name'])
            else:
                start = 0
            end_function = None
            hit_break = False
            for info_function in self.class_functions[start:]:
                end_function = info_function.quiz_function()
                if type(end_function) != tuple and end_function is True:
                    hit_break = True
                    break
                elif type(end_function) == tuple and end_function[0] is True:
                    end_function += (self.class_name, )
                    return end_function
        if hit_break is True:
            return False
        if end_function is None:
            return False
        elif end_function[len(end_function)-1] == self.class_functions[len(self.class_functions)-1].function_name:
            end_function += (self.class_name, )
            return end_function
        else:
            exit_tuple = (False, self.class_name)
            return exit_tuple

    def add_test_files_to_class(self, resume=None):
        print(Fore.MAGENTA + f'Class Name:\n\t{self.class_name}')
        choice = input(Fore.WHITE + 'Exit: e | Exit and Save: s | Continue Adding Test Files: enter key\n'
                       + Fore.LIGHTBLUE_EX + '\tChoose: ')
        if choice == 'e':
            return True
        if choice == 's':
            class_tuple = (True, self.class_name)
            return class_tuple
        else:
            if resume is not None and 'function_name' in resume:
                start = [x.function_name for x in self.class_functions].index(resume['function_name'])
            else:
                start = 0
            print(Fore.LIGHTMAGENTA_EX + f'Class Definition:\n\t' + '\n\t'.join(self.class_definition) + '\n\n')
            if not self.class_test_files:
                self.class_create_test_file()
            end_function = None
            hit_break = False
            for info_function in self.class_functions[start:]:
                end_function = info_function.add_test_files_to_func()
                if type(end_function) != tuple and end_function is True:
                    hit_break = True
                    break
                elif type(end_function) == tuple and end_function[0] is True:
                    end_function += (self.class_name,)
                    return end_function
        if hit_break is True:
            return False
        if end_function is None:
            return False
        elif end_function[len(end_function) - 1] == self.class_functions[len(self.class_functions) - 1].function_name:
            end_function += (self.class_name,)
            return end_function
        else:
            exit_tuple = (False, self.class_name)
            return exit_tuple

    def load_class_function_list(self, list_of_functions):
        for ind_function in list_of_functions:
            function_info = Function(ind_function['function_name'], ind_function['function_definition'])
            function_info.load_func_test_file_list(ind_function['function_test_files'])
            self.class_functions.append(function_info)

    def load_class_test_file_list(self, list_of_test_files):
        for ind_test_file in list_of_test_files:
            test_file_info = TestFile(ind_test_file['test_file_name'], ind_test_file['test_file_caption'])
            self.class_test_files.append(test_file_info)

    def write_program_class(self):
        program_class_dump = {
            'class_name':
                self.class_name,
            'class_definition':
                self.class_definition,
            'class_functions':
                [i.write_function() for i in self.class_functions],
            'class_test_files':
                [i.write_test_file() for i in self.class_test_files]
        }
        return program_class_dump

    def class_add_function(self):
        name_of_function = input(Fore.GREEN + '\t\tWhat is the name of the function you wish to add?\n\t\t')
        definition_of_function = textwrap.wrap(input(Fore.GREEN + f'\t\tWhat is the definition of {name_of_function}?\n\t\t'), 80)
        self.class_functions.append(Function(name_of_function, definition_of_function))

    def class_create_test_file(self):
        name_of_test_file = input(Fore.GREEN + '\t\tWhat is the name of the test file you wish to create?\n\t\t')
        caption_of_test_file = textwrap.wrap(input(Fore.GREEN + f'\t\tWhat is the caption of {name_of_test_file}?\n\t\t'), 80)
        self.class_test_files.append(TestFile(name_of_test_file, caption_of_test_file))

    def edit_class_function_info(self):
        for x in self.class_functions:
            print(Fore.YELLOW + x.function_name)
        function_choice = input(Fore.LIGHTBLUE_EX + "\tWhat is the name of the class you would like to edit?\n\t")
        self.class_functions[
            [x.function_name for x in self.class_functions].index(function_choice)
        ].function_internal_menu()

    def class_delete_function(self):
        name_of_function = input(Fore.RED + '\t\tWhat is the name of the function you wish to delete?\n\t\t')
        del self.class_functions[[x.function_name for x in self.class_functions].index(name_of_function)]

    def class_remove_test_file(self):
        name_of_test_file = input(Fore.RED + '\t\tWhat is the name of the test file you wish to remove?\n\t\t')
        del self.class_test_files[[x.test_file_name for x in self.class_test_files].index(name_of_test_file)]

    def class_function_menu(self):
        go = True
        while go:
            choice = input(Fore.YELLOW + f'Class Name: {self.class_name}\n\n' +
                           Fore.WHITE +
                           'Add Function: a | Delete Function: d | Create A Test File: c | Remove A Test File: r ' +
                           '| Edit Class Function Info: i | Exit: e\n\n'
                           + Fore.LIGHTBLUE_EX + '\tChoose: ')
            if choice == 'a':
                self.class_add_function()
            elif choice == 'd':
                self.class_delete_function()
            elif choice == 'c':
                self.class_create_test_file()
            elif choice == 'r':
                self.class_remove_test_file()
            elif choice == 'i':
                self.edit_class_function_info()
            elif choice == 'e':
                go = False
