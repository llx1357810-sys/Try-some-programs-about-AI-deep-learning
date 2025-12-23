# 0.5x² + 2x + 30,   宽度5  深度2   手动宽度   #后续任务1: 个性化跑二元一次 (1/1)  # 后续任务2: fixed width 变成其中2个可以自主选择细化 (0/1)
n = 0.0001
print(' Welcome to using this simple AI training show.    We have depth 2, width 5, width with fixed range |2|, x range [-5,5]')
print()
print('Now, we have a basic   Ax squared + Bx + C  . We will train AI to close to this function with only using \'ax + b\'')
A = float(input('please input the \' The coefficient of x squared\'?  (input number > 0)'))
B = float(input('please input the \' The coefficient of x \'?  (input number > 0)'))
C = float(input('please input the \' The constant \'?  <We recommend you input a large number >(input number > 0)'))

print()
time = int(input('please input how many times you want to train? (input nature number > 0)'))


import random
import sympy

x, y, error, loss = sympy.symbols('x y error loss')
a11,a12,b11,b12,a21,a22,b21,b22,a31,a32,b31,b32,a41,a42,b41,b42,a51,a52,b51,b52,a1,a2,b1,b2 = sympy.symbols('a11 a12 b11 b12 a21 a22 b21 b22 a31 a32 b31 b32 a41 a42 b41 b42 a51 a52 b51 b52 a1 a2 b1 b2')

dictionary = {a11:0.5,a12:0.5,b11:0.5,b12:0.5,a21:0.5,a22:0.5,b21:0.5,b22:0.5,a31:0.5,a32:0.5,b31:0.5,b32:0.5,a41:0.5,a42:0.5,b41:0.5,b42:0.5,a51:0.5,a52:0.5,b51:0.5,b52:0.5}

z1 = a1 * x + b1
z2 = a2 * z1 + b2
error = y - z2
loss = 0.5 * (error ** 2)


gradient_a1 = sympy.diff(loss,a1)
gradient_a2 = sympy.diff(loss,a2)
gradient_b1 = sympy.diff(loss,b1)
gradient_b2 = sympy.diff(loss,b2)


for i in range(0,time,1):
    x_now = round(random.uniform(-5,5),3)
    y_true = A * (x_now ** 2) + B * x_now + C

    if x_now < -3:
        a11_now = dictionary[a11]
        a12_now = dictionary[a12]
        b11_now = dictionary[b11]
        b12_now = dictionary[b12]

        dictionary_now = {a1:a11_now, a2:a12_now, b1:b11_now, b2:b12_now, x:x_now, y:y_true}

        z1_now = z1.subs(dictionary_now)
        z2_now = z2.subs(dictionary_now)
        error_now = error.subs(dictionary_now)
        loss_now = loss.subs(dictionary_now)
        
        gradient_a11_now = float(gradient_a1.subs(dictionary_now))
        gradient_a12_now = float(gradient_a2.subs(dictionary_now))
        gradient_b11_now = float(gradient_b1.subs(dictionary_now))
        gradient_b12_now = float(gradient_b2.subs(dictionary_now))

        a11_new = round(a11_now - n*gradient_a11_now, 8)
        a12_new = round(a12_now - n*gradient_a12_now, 8)
        b11_new = round(b11_now - n*gradient_b11_now, 8)
        b12_new = round(b12_now - n*gradient_b12_now, 8)

        dictionary[a11] = a11_new
        dictionary[a12] = a12_new
        dictionary[b11] = b11_new
        dictionary[b12] = b12_new

    elif x_now < -1:
        a21_now = dictionary[a21]
        a22_now = dictionary[a22]
        b21_now = dictionary[b21]
        b22_now = dictionary[b22]

        dictionary_now = {a1:a21_now, a2:a22_now, b1:b21_now, b2:b22_now, x:x_now, y:y_true}

        z1_now = z1.subs(dictionary_now)
        z2_now = z2.subs(dictionary_now)
        error_now = error.subs(dictionary_now)
        loss_now = loss.subs(dictionary_now)

        gradient_a21_now = float(gradient_a1.subs(dictionary_now))
        gradient_a22_now = float(gradient_a2.subs(dictionary_now))
        gradient_b21_now = float(gradient_b1.subs(dictionary_now))
        gradient_b22_now = float(gradient_b2.subs(dictionary_now))

        a21_new = round(a21_now - n*gradient_a21_now, 8)
        a22_new = round(a22_now - n*gradient_a22_now, 8)
        b21_new = round(b21_now - n*gradient_b21_now, 8)
        b22_new = round(b22_now - n*gradient_b22_now, 8)

        dictionary[a21] = a21_new
        dictionary[a22] = a22_new
        dictionary[b21] = b21_new
        dictionary[b22] = b22_new

    elif x_now < 1:
        a31_now = dictionary[a31]
        a32_now = dictionary[a32]
        b31_now = dictionary[b31]
        b32_now = dictionary[b32]

        dictionary_now = {a1:a31_now, a2:a32_now, b1:b31_now, b2:b32_now, x:x_now, y:y_true}

        z1_now = z1.subs(dictionary_now)
        z2_now = z2.subs(dictionary_now)
        error_now = error.subs(dictionary_now)
        loss_now = loss.subs(dictionary_now)
        
        gradient_a31_now = float(gradient_a1.subs(dictionary_now))
        gradient_a32_now = float(gradient_a2.subs(dictionary_now))
        gradient_b31_now = float(gradient_b1.subs(dictionary_now))
        gradient_b32_now = float(gradient_b2.subs(dictionary_now))

        a31_new = round(a31_now - n*gradient_a31_now, 8)
        a32_new = round(a32_now - n*gradient_a32_now, 8)
        b31_new = round(b31_now - n*gradient_b31_now, 8)
        b32_new = round(b32_now - n*gradient_b32_now, 8)

        dictionary[a31] = a31_new
        dictionary[a32] = a32_new
        dictionary[b31] = b31_new
        dictionary[b32] = b32_new

    elif x_now < 3:
        a41_now = dictionary[a41]
        a42_now = dictionary[a42]
        b41_now = dictionary[b41]
        b42_now = dictionary[b42]

        dictionary_now = {a1:a41_now, a2:a42_now, b1:b41_now, b2:b42_now, x:x_now, y:y_true}

        z1_now = z1.subs(dictionary_now)
        z2_now = z2.subs(dictionary_now)
        error_now = error.subs(dictionary_now)
        loss_now = loss.subs(dictionary_now)
        
        gradient_a41_now = float(gradient_a1.subs(dictionary_now))
        gradient_a42_now = float(gradient_a2.subs(dictionary_now))
        gradient_b41_now = float(gradient_b1.subs(dictionary_now))
        gradient_b42_now = float(gradient_b2.subs(dictionary_now))

        a41_new = round(a41_now - n*gradient_a41_now, 8)
        a42_new = round(a42_now - n*gradient_a42_now, 8)
        b41_new = round(b41_now - n*gradient_b41_now, 8)
        b42_new = round(b42_now - n*gradient_b42_now, 8)

        dictionary[a41] = a41_new
        dictionary[a42] = a42_new
        dictionary[b41] = b41_new
        dictionary[b42] = b42_new

    else:
        a51_now = dictionary[a51]
        a52_now = dictionary[a52]
        b51_now = dictionary[b51]
        b52_now = dictionary[b52]

        dictionary_now = {a1:a51_now, a2:a52_now, b1:b51_now, b2:b52_now, x:x_now, y:y_true}

        z1_now = z1.subs(dictionary_now)
        z2_now = z2.subs(dictionary_now)
        error_now = error.subs(dictionary_now)
        loss_now = loss.subs(dictionary_now)
        
        gradient_a51_now = float(gradient_a1.subs(dictionary_now))
        gradient_a52_now = float(gradient_a2.subs(dictionary_now))
        gradient_b51_now = float(gradient_b1.subs(dictionary_now))
        gradient_b52_now = float(gradient_b2.subs(dictionary_now))

        a51_new = round(a51_now - n*gradient_a51_now, 8)
        a52_new = round(a52_now - n*gradient_a52_now, 8)
        b51_new = round(b51_now - n*gradient_b51_now, 8)
        b52_new = round(b52_now - n*gradient_b52_now, 8)

        dictionary[a51] = a51_new
        dictionary[a52] = a52_new
        dictionary[b51] = b51_new
        dictionary[b52] = b52_new
    if i % 10000 == 0:
        print()
        print(f'Now, we have trained {i} times')
        print(f'The right is {round(y_true,3)}, the AI answer is {z2_now:.3f}.   (-5 =< x =<5 , now x = {round(x_now,3)})')
        print(f'print the dictionary is {dictionary}')
    elif i == time -1:
        print()
        print(f'Now, we have trained {time} times')
        print(f'The right is {round(y_true,3)}, the AI answer is {z2_now:.3f}.   (-5 =< x =<5 , now x = {round(x_now,3)})')
        print(f'print the dictionary is {dictionary}')
        print('We finish !')
