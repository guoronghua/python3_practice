num10 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
num20 = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
num100 = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


edit = [[i + j for j in range(5)] for i in range(3)]
print(edit)
[
 [0, 1, 2, 3, 4],
 [1, 2, 3, 4, 5],
 [2, 3, 4, 5, 6],
 [3, 4, 5, 6, 7],
 [4, 5, 6, 7, 8]
 ]