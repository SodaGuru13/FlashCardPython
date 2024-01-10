import os
import json
import textwrap
from colorama import Fore, Back, Style, init
init(autoreset=True)


def review(info):
    print()
    for key, explanation in info.items():
        os.system('cls')
        print('\033[1;34m' + f'{key}: ')
        flow = input('\033[0;31m' + 'Type End To End The Review Or Hit Enter To Continue? ')
        if flow == 'End':
            break
        print('\033[0;32m' + '\n'.join(explanation) + '\n')
        input()


def choose(info):
    key = input(Fore.LIGHTBLUE_EX + '\nEnter A Word: ')
    if key in info:
        print(Fore.LIGHTBLUE_EX + f'\n{key}: ')
        print(Fore.GREEN + '\n'.join(info[key]) + '\n')
        input()
    else:
        print(f'{key} is not in data_file.json.')
        input()


def add(info):
    word = input(Fore.LIGHTBLUE_EX + '\nEnter A Word: ')
    if word not in info:
        definition = textwrap.wrap(input(Fore.GREEN + 'Enter A Definition: '), 80)
        info[word] = definition
        myKeys = list(info.keys())
        myKeys.sort(key=str.lower)
        sortedList = {i: info[i] for i in myKeys}
        info = sortedList
    else:
        print('This word is already in data_file.json.')
        input()
    return info


def delete(info):
    key = input(Fore.RED + '\nEnter A Word: ')
    del info[key]
    return info


def quiz(info):
    print()
    for key, explanation in info.items():
        os.system('cls')
        print('\033[1;34m' + f'{key}: \n')
        flow = input('\033[0;31m' + 'Type End To End The Review Or Hit Enter To Continue? \n')
        if flow == 'End':
            break
        user_answer = input('\033[0;32mPlease enter the definition of the word:\n\n')
        if(user_answer.lower() == ' '.join(explanation).lower()):
            print('\n\033[0;32mCorrect answer\n')
        else:
            print('\n\033[0;31mWrong answer\n')
        print('\033[0;32m' + '\n'.join(explanation) + '\n')
        input()


def stop(go):
    go = False
    return go


def index():
    with open("data_file.json", "r") as read_file:
        info = json.load(read_file)
    go = True
    while go:
        os.system('cls')
        choice = input(Fore.WHITE + 'Review = r, Choose = c, Add = a, Delete = d, Quiz = q, or Exit = e: ')
        if choice == 'r':
            review(info)
        elif choice == 'c':
            choose(info)
        elif choice == 'a':
            info = add(info)
        elif choice == 'd':
            info = delete(info)
        elif choice == 'q':
            quiz(info)
        elif choice == 'e':
            go = stop(go)
    with open("data_file.json", "w") as write_file:
        json.dump(info, write_file, indent=6)


if __name__ == "__main__":
    index()