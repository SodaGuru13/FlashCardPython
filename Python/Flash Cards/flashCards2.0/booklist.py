from book import Book
import colorama
from colorama import Fore, Back, Style, init
init(autoreset=True)


class BookList:
    def __init__(self, book_list_name, book_list_author):
        self.book_list_name = book_list_name
        self.book_list_author = book_list_author
        self.books = []

    def display_book_list(self, display_type, resume=None):
        print(Fore.CYAN + f'Book List Title: {self.book_list_name.title()}\n' +
                          f'Book List Author: {self.book_list_author.title()}')
        if resume is not None and 'book_title' in resume:
            start = [x.book_title for x in self.books].index(resume['book_title'])
        else:
            start = 0
        end_book = None
        hit_break = False
        for info_book in self.books[start:]:
            if resume is None:
                end_book = info_book.display_book(display_type)
            else:
                end_book = info_book.display_book(display_type, resume)
            if type(end_book) != tuple and end_book is True:
                hit_break = True
                break
            elif type(end_book) == tuple and end_book[0] is True:
                end_book += (self.book_list_name, display_type)
                return end_book
        if hit_break is True:
            return False
        if end_book is None:
            return False
        elif type(end_book) == tuple and end_book[len(end_book)-1] == self.books[len(self.books)-1].book_title:
            end_book += (self.book_list_name, )
            return end_book
        else:
            return False

    def load_book_list(self, list_of_books):
        for ind_book in list_of_books:
            book_info = Book(ind_book['book_title'], ind_book['book_author'])
            book_info.load_chapter_list(ind_book['chapters'])
            self.books.append(book_info)

    def write_book_list(self):
        book_list_dict = {
            'book_list_name':
                self.book_list_name,
            'book_list_author':
                self.book_list_author,
            'books':
                [i.write_book() for i in self.books]
        }
        return book_list_dict

    def add_book(self):
        name_of_book = input(Fore.GREEN + '\t\tWhat is the title of the book you wish to add?\n\t\t')
        name_of_author = input(Fore.GREEN + f'\t\tWho is the author of {name_of_book}?\n\t\t')
        self.books.append(Book(name_of_book, name_of_author))

    def edit_book_info(self):
        for x in self.books:
            print(Fore.WHITE + x.book_title)
        book_choice = input(Fore.LIGHTBLUE_EX + "\tWhat is the title of the book you would like to edit?\n\t")
        self.books[[x.book_title for x in self.books].index(book_choice)].chapter_menu()

    def delete_book(self):
        title = input(Fore.RED + '\t\tWhat is the title of the book you wish to delete?\n\t\t')
        del self.books[[x.book_title for x in self.books].index(title)]

    def book_menu(self):
        go = True
        while go:
            choice = input(Fore.YELLOW +
                           f'Book List Title: {self.book_list_name.title()}\n' +
                           f'Book List Author: {self.book_list_author.title()}\n\n' +
                           Fore.WHITE + 'Add Book: a | Delete Book: d | Edit Book Info: i | Exit: e\n\n' +
                           Fore.LIGHTBLUE_EX + "\tChoose: ")
            if choice == 'a':
                self.add_book()
            elif choice == 'd':
                self.delete_book()
            elif choice == 'i':
                self.edit_book_info()
            elif choice == 'e':
                go = False
