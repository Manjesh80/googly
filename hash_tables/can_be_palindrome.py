from toolz import map, compose, frequencies

compute_frequency = compose(frequencies, list)
# odd_identifier = compose()

char_freq = compute_frequency(list("AABB"))
palindrome_possible = True
for k, v in char_freq.items():
    palindrome_possible = v % 2 == 0
    if palindrome_possible:
        break
print(f"Palindrome is {palindrome_possible}")
