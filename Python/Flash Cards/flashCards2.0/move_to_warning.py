import json
from booklist import BookList
from progclass import ProgramClass

def move_to_warning():
    with open("data_file_2.0.json", "r") as read_file:
        info = json.load(read_file)
    book_list = BookList(info['book_list_name'], info['book_list_author'])
    book_list.load_book_list(info['books'])
    classes_in_warning_list = []
    for alass in book_list.books[0].chapters[1].sections[3].section_classes:
        if alass.class_name.endswith('Warning'):
            book_list.books[0].chapters[1].sections[4].section_classes.append(
                ProgramClass(alass.class_name, alass.class_definition))
    for dlass in book_list.books[0].chapters[1].sections[4].section_classes:
        del book_list.books[0].chapters[1].sections[3].section_classes[
            [
                x.class_name for x in book_list.books[0].chapters[1].sections[3].section_classes
            ].index(dlass.class_name)]
    with open("data_file_2.0.json", "w") as write_file:
        json.dump(book_list.write_book_list(), write_file, indent=1)


if __name__ == '__main__':
    move_to_warning()