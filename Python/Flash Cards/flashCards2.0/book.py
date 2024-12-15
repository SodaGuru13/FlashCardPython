from chapter import Chapter
from colorama import Fore, Back, Style, init
init(autoreset=True)


class Book():
    def __init__(self, book_title, book_author):
        self.book_title = book_title
        self.book_author = book_author
        self.chapters = []

    def display_book(self, display_type, resume=None):
        print(Fore.LIGHTCYAN_EX + f'Book Title: {self.book_title.title()}\nBook Author: {self.book_author.title()}')
        choice = ''
        if display_type == 'r':
            choice = input(Fore.WHITE + 'Exit: e | Continue Reviewing Books: enter key\n' + Fore.LIGHTBLUE_EX +
                           '\tChoose: ')
        elif display_type == 'q':
            choice = input(Fore.WHITE + 'Exit: e | Continue Quiz Over Books: enter key\n' + Fore.LIGHTBLUE_EX +
                           '\tChoose: ')
        elif display_type == 't':
            choice = input(Fore.WHITE + 'Exit: e | Continue Adding Test Files to Books: enter key\n' +
                           Fore.LIGHTBLUE_EX + '\tChoose: ')
        if choice == 'e':
            return True
        else:
            if resume is not None and 'chapter_name' in resume:
                start = [x.chapter_name for x in self.chapters].index(resume['chapter_name'])
            else:
                start = 0
            end_chapter = None
            hit_break = False
            for info_chapter in self.chapters[start:]:
                if resume is None:
                    end_chapter = info_chapter.display_chapter(display_type)
                else:
                    end_chapter = info_chapter.display_chapter(display_type, resume)
                if type(end_chapter) != tuple and end_chapter is True:
                    hit_break = True
                    break
                elif type(end_chapter) == tuple and end_chapter[0] is True:
                    end_chapter += (self.book_title, )
                    return end_chapter
        if hit_break is True:
            return False
        if end_chapter is None:
            return False
        elif type(end_chapter) == tuple and end_chapter[len(end_chapter)-1] == self.chapters[len(self.chapters)-1].chapter_name:
            end_chapter += (self.book_title, )
            return end_chapter
        else:
            return False

    def load_chapter_list(self, list_of_chapters):
        for ind_chapter in list_of_chapters:
            chapter_info = Chapter(ind_chapter['chapter_name'])
            chapter_info.load_section_list(ind_chapter['sections'])
            self.chapters.append(chapter_info)

    def write_book(self):
        book_dump = {
            'book_title':
                self.book_title,
            'book_author':
                self.book_author,
            'chapters':
                [i.write_chapter() for i in self.chapters]
        }
        return book_dump

    def add_chapter(self):
        name_of_chapter = input(Fore.GREEN + '\t\tWhat is the name of the chapter you wish to add?\n\t\t')
        self.chapters.append(Chapter(name_of_chapter))

    def edit_chapter_info(self):
        for x in self.chapters:
            print(Fore.YELLOW + x.chapter_name)
        chapter_choice = input(Fore.LIGHTBLUE_EX + "\tWhat is the name of the chapter you would like to edit?\n\t")
        self.chapters[[x.chapter_name for x in self.chapters].index(chapter_choice)].section_menu()

    def delete_chapter(self):
        for x in self.chapters:
            print(x.chapter_name)
        name_of_chapter = input(Fore.RED + '\t\tWhat is the name of the chapter you wish to delete?\n\t\t')
        del self.chapters[[x.chapter_name for x in self.chapters].index(name_of_chapter)]

    def chapter_menu(self):
        go = True
        while go:
            choice = input(Fore.YELLOW +
                           f'Book Title: {self.book_title.title()}\nBook Author: {self.book_author.title()}\n\n' +
                           Fore.WHITE + 'Add Chapter: a | Delete Chapter: d: | Edit Chapter Info: i | Exit: e\n\n' +
                           Fore.LIGHTBLUE_EX + "\tChoose: ")
            if choice == 'a':
                self.add_chapter()
            elif choice == 'd':
                self.delete_chapter()
            elif choice == 'i':
                self.edit_chapter_info()
            elif choice == 'e':
                go = False
