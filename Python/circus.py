# Level 2: Part 1 Loops
	# word = 'Welcome!'
	# for letter in word:
		# print(letter)
# Level 2: Part 2 Loops
	# import random
	# for index in range(5):
		# print(random.randint(1,53))
# Level 2: Part 3 Loops
	# performances = {'Ventiloquism':'9:00am',
			# 'Snake Charmer': '12:00pm',
			# 'Amazing Acrobatics': '2:00pm',
			# 'Enchanted Elephants': '5:00pm'}
	# for name, value in performances.items():
		# print(name, ":", value, sep='')
# Level 2: Part 4 and 5 Loops
	# import random
	# num = random.randint(1, 10)
	# print(num, sep='')
	# guess = int(input('Guess a number between 1 and 10\n'))
	# times = 1
	# Add a while loop here
	# while guess != num:
		# guess = int(input('Guess again\n'))
		# times = times + 1
		# if times == 3:
			# break
	# if guess == num:
		# print('You Win!')
	# else:
		# print('You lose! The number was ' + str(num))
# Level 3: Part 1, 2, and 3
	# import random
	# def lotto_numbers():
	# 	# code in the function goes here
	# 	lotto_nums = []
	# 	for i in range(5):
	# 		lotto_nums.append(random.randint(1,53))
	# 	return lotto_nums
	#
	# 	# pass
	# numbers = lotto_numbers()
	# def main():
	# 	# code in the main function goes here
	# 	numbers = lotto_numbers()
	# 	print(numbers)
	#
	# main()
# Level 3: Part 4 and 5
	# import random
	# def guessing_game():
	# 	num = random.randint(1, 10)
	# 	guess = int(input('Guess a number between 1 and 10'))
	# 	times = 1
	# 	while guess != num:
	# 		guess = int(input('Guess again'))
	# 		times += 1
	# 		if times == 3:
	# 			break
	# 	if guess == num:
	# 		print('You win!')
	# 	else:
	# 		print('You lose! The number was', num)
	#
	# def lotto_numbers():
	# 	# code in the function goes here
	# 	lotto_nums = []
	# 	for i in range(5):
	# 		lotto_nums.append(random.randint(1,53))
	# 	return lotto_nums
	#
	# def main():
	# 	answer = input('Do you want to get lottery numbers (1) or play the game (2) or quit (Q)?')
	# 	if answer == '1':
	# 		numbers = lotto_numbers()
	# 		print(numbers)
	# 	elif answer == '2':
	# 		guessing_game()
	# 	else:
	# 		print('Toodles!')
	#
	# main()