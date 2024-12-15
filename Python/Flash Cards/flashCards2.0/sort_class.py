import textwrap
import json
from booklist import BookList
from book import Book
from chapter import Chapter
from section import Section
from progclass import ProgramClass
from function import Function


def sort_class():
    with open("data_file_2.0.json", "r") as read_file:
        info = json.load(read_file)
    book_list = BookList(info['book_list_name'], info['book_list_author'])
    book_list.load_book_list(info['books'])
    classes_in_function_list = []
    for func in book_list.books[0].chapters[0].sections[0].section_functions:
        if func.function_name.startswith('class ') | func.function_name.startswith('class.'):
            book_list.books[0].chapters[0].sections[0].section_classes.append(
                ProgramClass(func.function_name, func.function_definition))
    for lass in book_list.books[0].chapters[0].sections[0].section_classes:
        del book_list.books[0].chapters[0].sections[0].section_functions[
            [
                x.function_name for x in book_list.books[0].chapters[0].sections[0].section_functions
            ].index(lass.class_name)]
    with open("data_file_2.0.json", "w") as write_file:
        json.dump(book_list.write_book_list(), write_file, indent=1)


if __name__ == '__main__':
    sort_class()
