# Lease possible change page 29

coins = [1, 1, 1, 1, 1, 5, 10, 25]
max_possible_change = 0
for coin in coins:
    if coin > max_possible_change + 1:
        break
    max_possible_change += coin
print(f"Lowest not possible change is {max_possible_change+1}")
