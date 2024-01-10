import textwrap
from progclass import ProgramClass
from operator import itemgetter, attrgetter
from function import Function
from colorama import Fore, Back, Style, init
init(autoreset=True)


class Section:
    def __init__(self, section_name):
        self.section_name = section_name
        self.section_classes = []
        self.section_functions = []

    def display_section(self, display_type, resume=None):
        print(Fore.LIGHTYELLOW_EX + f'Section Title: {self.section_name.title()}')
        choice = ''
        if display_type == 'r':
            choice = input(Fore.WHITE + 'Exit: e | Continue Reviewing Sections: enter key\n' + Fore.LIGHTBLUE_EX +
                           '\tChoose: ')
        elif display_type == 'q':
            choice = input(Fore.WHITE + 'Exit: e | Continue Quiz Over Sections: enter key\n' + Fore.LIGHTBLUE_EX +
                           '\tChoose: ')
        elif display_type == 't':
            choice = input(Fore.WHITE + 'Exit: e | Continue Adding Test Files to Sections: enter key\n' +
                           Fore.LIGHTBLUE_EX + '\tChoose: ')
        if choice == 'e':
            return True
        else:
            review_class = True
            if resume is not None and 'class_name' in resume:
                start = [x.class_name for x in self.section_classes].index(resume['class_name'])
            elif resume is not None and 'function_name' in resume:
                review_class = False
                start = [x.function_name for x in self.section_functions].index(resume['function_name'])
            else:
                start = 0
            end_section_class = None
            hit_break = False
            if review_class is True:
                print('Section Classes: ')
                for info_class in self.section_classes[start:]:
                    if display_type == 'r':
                        if resume is None:
                            end_section_class = info_class.review_class()
                        else:
                            end_section_class = info_class.review_class(resume)
                    elif display_type == 'q':
                        if resume is None:
                            end_section_class = info_class.quiz_class()
                        else:
                            end_section_class = info_class.quiz_class(resume)
                    elif display_type == 't':
                        if resume is None:
                            end_section_class = info_class.add_test_files_to_class()
                        else:
                            end_section_class = info_class.add_test_files_to_class(resume)
                    if type(end_section_class) != tuple and end_section_class is True:
                        hit_break = True
                        break
                    elif type(end_section_class) == tuple and end_section_class[0] is True:
                        end_section_class += (self.section_name, )
                        return end_section_class
                start = 0
            if hit_break is True:
                exit_program_prompt = input(Fore.WHITE +
                                            'Would you like to exit here (type e) or continue (hit enter) to functions?'
                                            + '\n'
                                            + Fore.LIGHTBLUE_EX + '\tChoose: ')
                if exit_program_prompt == 'e':
                    return False
                else:
                    hit_break = False
            print('Section Functions: ')
            end_section_func = None
            for info_function in self.section_functions[start:]:
                if display_type == 'r':
                    end_section_func = info_function.review_function()
                elif display_type == 'q':
                    end_section_func = info_function.quiz_function()
                elif display_type == 't':
                    end_section_func = info_function.add_test_files_to_func()
                if type(end_section_func) != tuple and end_section_func is True:
                    hit_break = True
                    break
                elif type(end_section_func) == tuple and end_section_func[0] is True:
                    end_section_func += (self.section_name, )
                    return end_section_func
            if hit_break is True:
                return False
            elif end_section_func is not None and type(end_section_func) == tuple and \
                    end_section_func[len(end_section_func)-1] == \
                    self.section_functions[len(self.section_functions)-1].function_name:
                end_section_func += (self.section_name, )
                return end_section_func
            elif end_section_func is None and (end_section_class is not None or type(end_section_func) != tuple) \
                    and end_section_class[len(end_section_class)-1] == \
                    self.section_classes[len(self.section_classes)-1].class_name:
                end_section_class += (self.section_name, )
                return end_section_class
            else:
                return False

    def load_section_class_list(self, list_of_classes):
        for ind_class in list_of_classes:
            class_info = ProgramClass(ind_class['class_name'], ind_class['class_definition'])
            class_info.load_class_function_list(ind_class['class_functions'])
            class_info.load_class_test_file_list(ind_class['class_test_files'])
            self.section_classes.append(class_info)

    def load_section_function_list(self, list_of_functions):
        for ind_function in list_of_functions:
            function_info = Function(ind_function['function_name'], ind_function['function_definition'])
            function_info.load_func_test_file_list(ind_function['function_test_files'])
            self.section_functions.append(function_info)

    def write_section(self):
        section_dump = {
            'section_name':
                self.section_name,
            'section_classes':
                [i.write_program_class() for i in self.section_classes],
            'section_functions':
                [i.write_function() for i in self.section_functions]
        }
        return section_dump

    def section_add_class(self):
        name_of_class = input(Fore.GREEN + '\t\tWhat is the name of the class you wish to add?\n\t\t')
        definition_of_class = textwrap.wrap(input(Fore.GREEN +
                                                  f'\t\tWhat is the definition of {name_of_class}?\n\t\t'), 80)
        self.section_classes.append(ProgramClass(name_of_class, definition_of_class))
        self.section_classes.sort(key=lambda lass: lass.class_name.lower())

    def section_add_function(self):
        name_of_function = input(Fore.GREEN + '\t\tWhat is the name of the function you wish to add?\n\t\t')
        definition_of_function = textwrap.wrap(input(Fore.GREEN +
                                                     f'\t\tWhat is the definition of {name_of_function}?\n\t\t'), 80)
        self.section_functions.append(Function(name_of_function, definition_of_function))
        self.section_functions.sort(key=lambda func: func.function_name.lower())

    def edit_section_class_info(self):
        for x in self.section_classes:
            print(Fore.YELLOW + x.class_name)
        class_choice = input(Fore.LIGHTBLUE_EX + "\tWhat is the name of the class you would like to edit?\n\t")
        self.section_classes[[x.class_name for x in self.section_classes].index(class_choice)].class_function_menu()

    def edit_section_function_info(self):
        for x in self.section_functions:
            print(Fore.YELLOW + x.function_name)
        function_choice = input(Fore.LIGHTBLUE_EX + "\tWhat is the name of the class you would like to edit?\n\t")
        self.section_functions[
            [x.function_name for x in self.section_functions].index(function_choice)
        ].function_internal_menu()

    def section_delete_class(self):
        name_of_class = input(Fore.RED + '\t\tWhat is the name of the class you wish to delete?\n\t\t')
        del self.section_classes[[x.class_name for x in self.section_classes].index(name_of_class)]

    def section_delete_function(self):
        name_of_function = input(Fore.RED + '\t\tWhat is the name of the function you wish to delete?\n\t\t')
        del self.section_functions[[x.function_name for x in self.section_functions].index(name_of_function)]

    def section_class_menu(self):
        go = True
        while go:
            choice = input(Fore.YELLOW + f'Section Name: {self.section_name.title()}\n\n' +
                           Fore.WHITE + 'Add Class: a | Delete Class: d | Edit Section Class Info: i | Exit: e\n\n' +
                           Fore.LIGHTBLUE_EX + '\tChoose: ')
            if choice == 'a':
                self.section_add_class()
            elif choice == 'd':
                self.section_delete_class()
            elif choice == 'i':
                self.edit_section_class_info()
            elif choice == 'e':
                go = False

    def section_function_menu(self):
        go = True
        while go:
            choice = input(Fore.YELLOW + f'Section Name: {self.section_name.title()}\n\n' + Fore.WHITE +
                           'Add Function: a | Delete Function: d | Edit Section Function Info: i | Exit: e\n\n' +
                           Fore.LIGHTBLUE_EX + '\tChoose: ')
            if choice == 'a':
                self.section_add_function()
            elif choice == 'd':
                self.section_delete_function()
            elif choice == 'i':
                self.edit_section_function_info()
            elif choice == 'e':
                go = False

    def section_control_menu(self):
        go = True
        while go:
            choice = input(Fore.YELLOW + f'Section Name: {self.section_name.title()}\n\n' +
                           Fore.WHITE + 'Edit Functions: f | Edit Classes: c | Exit: e\n\n' +
                           Fore.LIGHTBLUE_EX + '\tChoose: ')
            if choice == 'f':
                self.section_function_menu()
            elif choice == 'c':
                self.section_class_menu()
            elif choice == 'e':
                go = False
