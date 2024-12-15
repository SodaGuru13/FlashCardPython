# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')   Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
    # print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# performances = {'Ventriloquism':    '9:00am',
#                 'Snake Charmer':    '12:00pm',
#                 'Amazing Acrobatics': '2:00pm',
#                 'Enchanted Elephants': '5:00pm'}
# schedule_file = open('schedule.txt', 'w')
#
# for key, val in performances.items():
#     schedule_file.write(key + '-' + val + '\n')
#
# schedule_file.close()

# schedule_file = open('schedule.txt', 'r')
#
# for line in schedule_file:
#     print(line)
#
# schedule_file.close()
# schedule_file = open('schedule.txt', 'r')
# for line in schedule_file:
#     (show, time) = line.strip().split('-')
#     print(show, time)
#
# schedule_file.close()

# performances = {}
# schedule_file = open('schedule.txt', 'r')
# for line in schedule_file:
#     (show, time) = line.split('-')
#     print(show, time)
#     performances[show] = time.strip()
#
# schedule_file.close()
# print(performances)
# schedule_file.close()

# performances = {}
#
# try:
#     schedule_file = open('schedule1.txt', 'r')
# except FileNotFoundError as err:
#     print(err)
#     exit(1)
#
# for line in schedule_file:
#     (show, time) = line.split('-')
#     print(show, time)
#     performances[show] = time.strip()
#
# schedule_file.close()
# print(performances)

# import requests
# url = "http://api.openweathermap.org/data/2.5/weather?q=Sioux%20Falls&units=imperial&APPID=60c61be3700d4eb3171fa87fc11670c3"
# request = requests.get(url)
#
# weather_json = request.json()
# print(weather_json)
# weather_main = weather_json['main']
#
# temp = weather_main['temp']
# print("The Circus's current temperature is ", temp)

# import weather
# url = "http://api.openweathermap.org/data/2.5/weather?q=Orlando&units=imperial&APPID=60c61be3700d4eb3171fa87fc11670c3"
# def main():
#     weather.current_weather()
#
# main()

