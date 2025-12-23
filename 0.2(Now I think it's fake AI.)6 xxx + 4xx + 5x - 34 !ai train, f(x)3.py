# 6 x**3 + 4 x**2 + 5 x - 34

import random
import sympy
x,y = sympy.symbols('x y')
a1,a2,a3,b1,b2,b3 = sympy.symbols('a1 a2 a3 b1 b2 b3')

dictionary = {a1:0.5,a2:0.5,a3:0.5,b1:0.5,b2:0.5,b3:0.5,}
n = 0.000000001

z1 = a1 * x + b1
z2 = a2 * z1 * z1 + b2
z3 = a3 + z2 * z2 * z2 + b3
error = y - z3
loss = 0.5 * error * error

gradient_a1 = sympy.diff(loss,a1)
gradient_a2 = sympy.diff(loss,a2)
gradient_a3 = sympy.diff(loss,a3)
gradient_b1 = sympy.diff(loss,b1)
gradient_b2 = sympy.diff(loss,b2)
gradient_b3 = sympy.diff(loss,b3)

time =int(input('train how many time'))
for i in range(0,time,1):
    x_now = random.randint(-10,10)
    y_now = 6 * (x_now ** 3) + 4 * (x_now ** 2) + 5 * x_now - 34
    a1_now = dictionary[a1]
    a2_now = dictionary[a2]
    a3_now = dictionary[a3]
    b1_now = dictionary[b1]
    b2_now = dictionary[b2]
    b3_now = dictionary[b3]

    new_dictionary = {a1:a1_now,a2:a2_now,a3:a3_now,b1:b1_now,b2:b2_now,b3:b3_now,x:x_now,y:y_now}
    z1_now = z1.subs(new_dictionary)
    z2_now = z2.subs(new_dictionary)
    z3_now = z3.subs(new_dictionary)
    error_now = error.subs(new_dictionary)
    loss_now = loss.subs(new_dictionary)

    gradient_a1_now = gradient_a1.subs(new_dictionary)
    gradient_a2_now = gradient_a2.subs(new_dictionary)
    gradient_a3_now = gradient_a3.subs(new_dictionary)
    gradient_b1_now = gradient_b1.subs(new_dictionary)
    gradient_b2_now = gradient_b2.subs(new_dictionary)
    gradient_b3_now = gradient_b3.subs(new_dictionary)

    a1_new = a1_now - n * gradient_a1_now
    a2_new = a2_now - n * gradient_a2_now
    a3_new = a3_now - n * gradient_a3_now
    b1_new = b1_now - n * gradient_b1_now
    b2_new = b2_now - n * gradient_b2_now
    b3_new = b3_now - n * gradient_b3_now

    dictionary[a1] = round(float(a1_new),9)
    dictionary[a2] = round(float(a2_new),9)
    dictionary[a3] = round(float(a3_new),9)
    dictionary[b1] = round(float(b1_new),9)
    dictionary[b2] = round(float(b2_new),9)
    dictionary[b3] = round(float(b3_new),9)

    if i % 10000 == 0 or i == time - 1:
        print()
        print(f'time is {i}, right answer {y_now:.3f}, Ai answer {z3_now:.3f}.   count:{dictionary}')
