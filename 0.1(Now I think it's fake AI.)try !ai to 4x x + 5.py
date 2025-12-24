# to make it as 4x**2 + 5  after 10000times:{a1: 2.000490498446, a2: 2.002662106255, b1: 0.508721999252, b2: 0.505462543587}
import sympy
import random
x,y = sympy.symbols('x y')
a1,b1,a2,b2 = sympy.symbols('a1 b1 a2 b2')

dictionary0 = {a1: 0.5 ,a2: 0.5 ,b1: 0.5 ,b2: 0.5}
dictionary ={a1: 2.000490498446, a2: 2.002662106255, b1: 0.508721999252, b2: 0.505462543587}
n = 0.00000001

z1 = a1 * x * x + b1
z2 = a2 * z1 + b2

error = y - z2
loss = 0.5 * error * error

tidu_a1 = sympy.diff(loss,a1)
tidu_b1 = sympy.diff(loss,b1)
tidu_a2 = sympy.diff(loss,a2)
tidu_b2 = sympy.diff(loss,b2)

time = int(input('input how many times'))
for i in range(0,time,1):
    a1_now = dictionary[a1]
    b1_now = dictionary[b1]
    a2_now = dictionary[a2]
    b2_now = dictionary[b2]
    x_now = random.randint(-30,30)
    y_now = 4 * x_now * x_now + 5
    
    dictionary_now = {x: x_now,y:y_now,a1:a1_now,b1:b1_now,a2:a2_now,b2:b2_now}
    error_now = error.subs(dictionary_now)
    loss_now = loss.subs(dictionary_now)
    z1_now = z1.subs(dictionary_now)
    z2_now = z2.subs(dictionary_now)

    grad_a1 = tidu_a1.subs(dictionary_now)
    grad_b1 = tidu_b1.subs(dictionary_now)
    grad_a2 = tidu_a2.subs(dictionary_now)
    grad_b2 = tidu_b2.subs(dictionary_now)

    a1_new = a1_now - n * grad_a1
    b1_new = b1_now - n * grad_b1
    a2_new = a2_now - n * grad_a2
    b2_new = b2_now - n * grad_b2

    dictionary[a1] = round(float(a1_new),12)
    dictionary[b1] = round(float(b1_new),12)
    dictionary[a2] = round(float(a2_new),12)
    dictionary[b2] = round(float(b2_new),12)

    if i % 10000 == 0 or i == time - 1:
        y_copy = round(float(4 * x_now * x_now + 5),3)
        z2_copy = round(float(z2.subs(dictionary_now)),3)
        print(f'time is {i}:')
        print(f'the True answer = {y_copy}, the AI answer = {z2_copy}, the dictionary:{dictionary}')

