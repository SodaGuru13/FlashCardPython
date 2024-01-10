from booklist import BookList
import json
import os
import colorama
from colorama import Fore, Back, Style, init
init(autoreset=True)


def create_book_list():
    name_of_book_list = input('What is the name of the book list you wish to create?')
    author_of_book_list = input(f'Who is the author of {name_of_book_list}?')
    book_list = BookList(name_of_book_list, author_of_book_list)
    return book_list


def destroy_book_list(book_list):
    del book_list
    return None


def save_function(save_data, default_return):
    if type(save_data) == tuple and save_data[0] is True:
        save_dict = {}
        save_list_outline = ['function_name', 'class_name', 'section_name', 'chapter_name', 'book_title',
                             'book_list_name', 'display_type']
        if len(save_data) - 1 == len(save_list_outline) - 1:
            if save_data[1].startswith('class ') or save_data[1].startswith('class.'):
                save_dict[save_list_outline[1]] = save_data[1]
            else:
                save_dict[save_list_outline[0]] = save_data[1]
            for i in reversed(save_data[2:]):
                save_dict[save_list_outline[save_data.index(i)]] = i
        elif len(save_data) - 1 == len(save_list_outline):
            for i in reversed(save_data[1:]):
                save_dict[save_list_outline[save_data.index(i)-1]] = i
        with open("save_spot.json", "w") as write_file:
            json.dump(save_dict, write_file, indent=1)
        return save_dict
    elif type(save_data) == tuple and save_data[0] is False:
        with open("save_spot.json", "w") as write_file:
            write_file.truncate()
        return {}
    else:
        return default_return


def main():
    check_file = os.path.getsize("data_file_2.0.json")
    if check_file == 0:
        book_list = create_book_list()
    else:
        with open("data_file_2.0.json", "r") as read_file:
            info = json.load(read_file)
        book_list = BookList(info['book_list_name'], info['book_list_author'])
        book_list.load_book_list(info['books'])
    check_save_state_file = os.path.getsize("save_spot.json")
    if check_save_state_file == 0 or check_file == 0:
        resume = {}
    else:
        with open("save_spot.json", "r") as read_file:
            resume = json.load(read_file)
    save = ()
    go = True
    while go:
        if len(resume) > 0:
            choice = input(Fore.WHITE + 'Edit Book List: m | Delete Book List: d | Review Book List: r ' +
                                        '| Quiz Book List: q | Back Up Save: b | Resume Review or Quiz: g ' +
                                        '| Exit: e\n\n' +
                           Fore.LIGHTBLUE_EX + "\tChoose: ")
        else:
            choice = input(Fore.WHITE + 'Edit Book List: m | Delete Book List: d | Review Book List: r ' +
                                        '| Quiz Book List: q | Add A Single Test File to Every class and Function: t ' +
                                        '| Back Up Save: b | Exit: e\n\n' +
                           Fore.LIGHTBLUE_EX + "\tChoose: ")
        if choice == 'm':
            book_list.book_menu()
        elif choice == 'd':
            book_list = destroy_book_list(book_list)
        elif choice == 'r':
            save = book_list.display_book_list(choice)
            resume = save_function(save, resume)
        elif choice == 'q':
            save = book_list.display_book_list(choice)
            resume = save_function(save, resume)
        elif choice == 't':
            save = book_list.display_book_list(choice)
            resume = save_function(save, resume)
        elif choice == 'b':
            with open("backup_file.json", "w") as write_file:
                json.dump(book_list.write_book_list(), write_file, indent=1)
        if choice == 'g':
            choice = resume['display_type']
            save = book_list.display_book_list(choice, resume)
            resume = save_function(save, resume)
        elif choice == 'e':
            go = False
    if book_list is None:
        with open("data_file_2.0.json", "w") as write_file:
            write_file.write('')
    else:
        with open("data_file_2.0.json", "w") as write_file:
            json.dump(book_list.write_book_list(), write_file, indent=1)


if __name__ == '__main__':
    main()
