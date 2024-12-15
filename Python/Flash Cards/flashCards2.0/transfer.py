import textwrap
import json
from booklist import BookList
from book import Book
from chapter import Chapter
from section import Section
from function import Function


def transfer():
    with open("../data_file.json", "r") as read_file:
        info_dump = json.load(read_file)
    book_list = BookList("Flash Card Book List", "Clayton Buus")
    book_list.books.append(Book("The Python Standard Library", "Python Software Foundation"))
    book_list.books[0].chapters.append(Chapter("Introduction through Built-in Types"))
    book_list.books[0].chapters[0].sections.append(
        Section("Notes On Availability through Integer String Conversion Length Limitation"))
    for key, explanation in info_dump.items():
        book_list.books[0].chapters[0].sections[0].section_functions.append(
            Function(key, textwrap.wrap(' '.join(explanation), 80)))
    with open("data_file_2.0.json", "w") as write_file:
        json.dump(book_list.write_book_list(), write_file, indent=1)


if __name__ == '__main__':
    transfer()