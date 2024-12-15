from section import Section
from colorama import Fore, Back, Style, init
init(autoreset=True)

class Chapter:
    def __init__(self, chapter_name):
        self.chapter_name = chapter_name
        self.sections = []

    def display_chapter(self, display_type, resume=None):
        print(Fore.YELLOW + f'Chapter Title: {self.chapter_name.title()}')
        choice = ''
        if display_type == 'r':
            choice = input(Fore.WHITE + 'Exit: e | Continue Reviewing Chapters: enter key\n' + Fore.LIGHTBLUE_EX +
                           '\tChoose: ')
        elif display_type == 'q':
            choice = input(Fore.WHITE + 'Exit: e | Continue Quiz Over Chapters: enter key\n' + Fore.LIGHTBLUE_EX +
                           '\tChoose: ')
        elif display_type == 't':
            choice = input(Fore.WHITE + 'Exit: e | Continue Adding Test Files to Chapters: enter key\n' +
                           Fore.LIGHTBLUE_EX + '\tChoose: ')
        if choice == 'e':
            return True
        else:
            if resume is not None and 'section_name' in resume:
                start = [x.section_name for x in self.sections].index(resume['section_name'])
            else:
                start = 0
            end_section = None
            hit_break = False
            for info_section in self.sections[start:]:
                if resume is None:
                    end_section = info_section.display_section(display_type)
                else:
                    end_section = info_section.display_section(display_type, resume)
                if type(end_section) != tuple and end_section is True:
                    hit_break = True
                    break
                elif type(end_section) == tuple and end_section[0] is True:
                    end_section += (self.chapter_name, )
                    return end_section
        if hit_break is True:
            return False
        if end_section is None:
            return False
        elif type(end_section) == tuple and end_section[len(end_section)-1] == self.sections[len(self.sections)-1].section_name:
            end_section += (self.chapter_name, )
            return end_section
        else:
            return False

    def load_section_list(self, list_of_sections):
        for ind_section in list_of_sections:
            section_info = Section(ind_section['section_name'])
            section_info.load_section_class_list(ind_section['section_classes'])
            section_info.load_section_function_list(ind_section['section_functions'])
            self.sections.append(section_info)

    def write_chapter(self):
        chapter_dict = {
            'chapter_name':
                self.chapter_name,
            'sections':
                [i.write_section() for i in self.sections]
        }
        return chapter_dict

    def add_section(self):
        name_of_section = input(Fore.GREEN + '\t\tWhat is the name of the section you wish to add?\n\t\t')
        self.sections.append(Section(name_of_section))

    def edit_chapter_info(self):
        for x in self.sections:
            print(Fore.YELLOW + x.section_name)
        section_choice = input(Fore.LIGHTBLUE_EX + "\tWhat is the name of the section you would like to edit?\n\t")
        self.sections[[x.section_name for x in self.sections].index(section_choice)].section_control_menu()

    def delete_section(self):
        name_of_section = input(Fore.RED + '\t\tWhat is the name of the section you wish to delete?\n\t\t')
        del self.sections[[x.section_name for x in self.sections].index(name_of_section)]

    def section_menu(self):
        go = True
        while go:
            choice = input(Fore.YELLOW + f'Chapter Title: {self.chapter_name.title()}\n\n' +
                           Fore.WHITE + 'Add Section: a | Delete Section: d | Edit Section Info: i | Exit: e\n\n' +
                           Fore.LIGHTBLUE_EX + '\tChoose: ')
            if choice == 'a':
                self.add_section()
            elif choice == 'd':
                self.delete_section()
            elif choice == 'i':
                self.edit_chapter_info()
            elif choice == 'e':
                go = False
