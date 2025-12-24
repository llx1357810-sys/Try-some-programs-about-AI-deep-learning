# 0.5x² + 2x + 30, Width 5, Depth 2, Manual Width # Follow-up Task 1: Personalize the binary representation once (1/1) # Follow-up Task 2: Change the fixed width to 5 weights that can be refined independently (5/5)
# 0.5x² + 2x + 30,   宽度5  深度2   手动宽度   #后续任务1: 个性化跑二元一次 (1/1)  # 后续任务2: fixed width 变成5个weight可以自主选择细化 (5/5)

# But this progrom have some big problems. Such as weight is not be limited. It can be become negative number (I have drawed some x-y picture and I believe : When a weight become form positive to negative, the answer_line will not become that we want, it not decrease gradient or the height, it will like a 'W')
# And, this 'ai' always study too snowly, so in 2.0, I will change the 'n'(study level) and make sure this ai will not over_study.
# In really ai, it can droupout in training, I will try it come true in my progrom.

n = 0.0001
print(' Welcome to using this simple AI training show.    We have depth 2, width 5, width with fixed range |2|, x range [-5,5]')
print()
print('Now, we have a basic   Ax² + Bx + C  . We will train AI to close to this function with only using \'ax + b\'')
A = float(input('please input the \' The coefficient of x²\'?  (input number > 0)'))
B = float(input('please input the \' The coefficient of x \'?  (input number > 0)'))
C = float(input('please input the \' The constant \'?  <We recommend you input a large number >(input number > 0)'))

#print()
time = int(input('please input how many times you want to train? (input nature number > 0)'))


import random
import sympy
import copy

x, y, error, loss = sympy.symbols('x y error loss')
w1, w2, w3, w4, w5,w_d_1,w_d_2,w_d_3,w_d_4,w_d_5 = sympy.symbols('w1 w2 w3 w4 w5 w_d_1 w_d_2 w_d_3 w_d_4 w_d_5')
a11,a12,b11,b12,a21,a22,b21,b22,a31,a32,b31,b32,a41,a42,b41,b42,a51,a52,b51,b52,a1,a2,b1,b2,weight = sympy.symbols('a11 a12 b11 b12 a21 a22 b21 b22 a31 a32 b31 b32 a41 a42 b41 b42 a51 a52 b51 b52 a1 a2 b1 b2 weight')

dictionary = {a11:0.5,a12:0.5,b11:0.5,b12:0.5,a21:0.5,a22:0.5,b21:0.5,b22:0.5,a31:0.5,a32:0.5,b31:0.5,b32:0.5,a41:0.5,a42:0.5,b41:0.5,b42:0.5,a51:0.5,a52:0.5,b51:0.5,b52:0.5,w1:1,w2:1,w3:1,w4:1,w5:1}

ai_answer = (a12 * (a11 * x + b11) + b12) * w1 * w_d_1 + (a22 * (a21 * x + b21) + b12) * w2 * w_d_2 + (a32 * (a31 * x + b31) + b32) * w3 * w_d_3 + (a42 * (a41 * x + b41) + b42) * w4 * w_d_4 + (a52 * (a51 * x + b51) + b52) * w5 * w_d_5



z1 = a1 * x + b1
z2 = (a2 * z1 + b2) * weight
error = y - ai_answer
loss = 0.5 * (error ** 2)



gradient_w1 = sympy.diff(loss, w1)
gradient_w2 = sympy.diff(loss, w2)
gradient_w3 = sympy.diff(loss, w3)
gradient_w4 = sympy.diff(loss, w4)
gradient_w5 = sympy.diff(loss, w5)


for i in range(0,time,1):
    x_now = round(random.uniform(-5,5),3)
    y_true =float( A * (x_now ** 2) + B * x_now + C )

#main caclulation
    d_gap1 = round((1 / (-5 - x_now + 0.0000001)) ** 2,8)
    d_gap2 = round((1 / (-2.5 - x_now + 0.0000001)) ** 2,8)
    d_gap3 = round((1 / (0 - x_now + 0.0000001)) ** 2,8)
    d_gap4 = round((1 / (2.5 - x_now + 0.0000001)) ** 2,8)
    d_gap5 = round((1 / (5 - x_now + 0.0000001)) ** 2,8)

    total_d_gap = d_gap1 + d_gap2 + d_gap3 + d_gap4 + d_gap5
    w_d_1_value = d_gap1 / total_d_gap
    w_d_2_value = d_gap2 / total_d_gap
    w_d_3_value = d_gap3 / total_d_gap
    w_d_4_value = d_gap4 / total_d_gap
    w_d_5_value = d_gap5 / total_d_gap
    sub_dictionary = copy.deepcopy(dictionary)
    sub_dictionary[w_d_1] = w_d_1_value
    sub_dictionary[w_d_2] = w_d_2_value
    sub_dictionary[w_d_3] = w_d_3_value
    sub_dictionary[w_d_4] = w_d_4_value
    sub_dictionary[w_d_5] = w_d_5_value
    sub_dictionary[x] = x_now
    
    ai_answer_now = ai_answer.subs(sub_dictionary)
    
    
 #first width
    a11_now = dictionary[a11]
    a12_now = dictionary[a12]
    b11_now = dictionary[b11]
    b12_now = dictionary[b12]
    weight_now = dictionary[w1]

    dictionary_now = {a11:a11_now, a12:a12_now, b11:b11_now, b12:b12_now, x:x_now, y:y_true,weight:weight_now}
    for key, term in sub_dictionary.items():
        if key not in dictionary_now:
            dictionary_now[key] = 0
        dictionary_now[key] = term
                
    error_now = error.subs(dictionary_now)
    loss_now = loss.subs(dictionary_now)

    gradient_a1 = sympy.diff(loss,a11)
    gradient_a2 = sympy.diff(loss,a12)
    gradient_b1 = sympy.diff(loss,b11)
    gradient_b2 = sympy.diff(loss,b12)
        
    gradient_a11_now = float(gradient_a1.subs(dictionary_now))
    gradient_a12_now = float(gradient_a2.subs(dictionary_now))
    gradient_b11_now = float(gradient_b1.subs(dictionary_now))
    gradient_b12_now = float(gradient_b2.subs(dictionary_now))
    gradient_weight_now = float(gradient_w1.subs(dictionary_now))

    a11_new = round(a11_now - n*gradient_a11_now, 8)
    a12_new = round(a12_now - n*gradient_a12_now, 8)
    b11_new = round(b11_now - n*gradient_b11_now, 8)
    b12_new = round(b12_now - n*gradient_b12_now, 8)
    weight_new = round(weight_now - n*gradient_weight_now, 8)

    dictionary[a11] = a11_new
    dictionary[a12] = a12_new
    dictionary[b11] = b11_new
    dictionary[b12] = b12_new
    dictionary[w1] = weight_new

#second width

    a21_now = dictionary[a21]
    a22_now = dictionary[a22]
    b21_now = dictionary[b21]
    b22_now = dictionary[b22]
    weight_now = dictionary[w2]

    dictionary_now = {a21:a21_now, a22:a22_now, b21:b21_now, b22:b22_now, x:x_now, y:y_true, weight:weight_now}
    for key, term in sub_dictionary.items():
        if key not in dictionary_now:
            dictionary_now[key] = 0
        dictionary_now[key] = term

    error_now = error.subs(dictionary_now)
    loss_now = loss.subs(dictionary_now)
    
    gradient_a21 = sympy.diff(loss,a21)
    gradient_a22 = sympy.diff(loss,a22)
    gradient_b21 = sympy.diff(loss,b21)
    gradient_b22 = sympy.diff(loss,b22)
    gradient_a21_now = float(gradient_a1.subs(dictionary_now))
    gradient_a22_now = float(gradient_a2.subs(dictionary_now))
    gradient_b21_now = float(gradient_b1.subs(dictionary_now))
    gradient_b22_now = float(gradient_b2.subs(dictionary_now))
    gradient_weight_now = float(gradient_w2.subs(dictionary_now))

    a21_new = round(a21_now - n*gradient_a21_now, 8)
    a22_new = round(a22_now - n*gradient_a22_now, 8)
    b21_new = round(b21_now - n*gradient_b21_now, 8)
    b22_new = round(b22_now - n*gradient_b22_now, 8)
    weight_new = round(weight_now - n*gradient_weight_now, 8)

    dictionary[a21] = a21_new
    dictionary[a22] = a22_new
    dictionary[b21] = b21_new
    dictionary[b22] = b22_new
    dictionary[w2] = weight_new


#third width

    a31_now = dictionary[a31]
    a32_now = dictionary[a32]
    b31_now = dictionary[b31]
    b32_now = dictionary[b32]
    weight_now = dictionary[w3]

    dictionary_now = {a31:a31_now, a32:a32_now, b31:b31_now, b32:b32_now, x:x_now, y:y_true, weight:weight_now}
    for key, term in sub_dictionary.items():
        if key not in dictionary_now:
            dictionary_now[key] = 0
        dictionary_now[key] = term

    error_now = error.subs(dictionary_now)
    loss_now = loss.subs(dictionary_now)

    gradient_a31 = sympy.diff(loss,a31)
    gradient_a32 = sympy.diff(loss,a32)
    gradient_b31 = sympy.diff(loss,b31)
    gradient_b32 = sympy.diff(loss,b32)
    gradient_a31_now = float(gradient_a1.subs(dictionary_now))
    gradient_a32_now = float(gradient_a2.subs(dictionary_now))
    gradient_b31_now = float(gradient_b1.subs(dictionary_now))
    gradient_b32_now = float(gradient_b2.subs(dictionary_now))
    gradient_weight_now = float(gradient_w3.subs(dictionary_now))

    a31_new = round(a31_now - n*gradient_a31_now, 8)
    a32_new = round(a32_now - n*gradient_a32_now, 8)
    b31_new = round(b31_now - n*gradient_b31_now, 8)
    b32_new = round(b32_now - n*gradient_b32_now, 8)
    weight_new = round(weight_now - n*gradient_weight_now, 8)

    dictionary[a31] = a31_new
    dictionary[a32] = a32_new
    dictionary[b31] = b31_new
    dictionary[b32] = b32_new
    dictionary[w3] = weight_new


#fourth width

    a41_now = dictionary[a41]
    a42_now = dictionary[a42]
    b41_now = dictionary[b41]
    b42_now = dictionary[b42]
    weight_now = dictionary[w4]

    dictionary_now = {a41:a41_now, a42:a42_now, b41:b41_now, b42:b42_now, x:x_now, y:y_true, weight:weight_now}
    for key, term in sub_dictionary.items():
        if key not in dictionary_now:
            dictionary_now[key] = 0
        dictionary_now[key] = term

    error_now = error.subs(dictionary_now)
    loss_now = loss.subs(dictionary_now)

    gradient_a41 = sympy.diff(loss,a41)
    gradient_a42 = sympy.diff(loss,a42)
    gradient_b41 = sympy.diff(loss,b41)
    gradient_b42 = sympy.diff(loss,b42)
    gradient_a41_now = float(gradient_a1.subs(dictionary_now))
    gradient_a42_now = float(gradient_a2.subs(dictionary_now))
    gradient_b41_now = float(gradient_b1.subs(dictionary_now))
    gradient_b42_now = float(gradient_b2.subs(dictionary_now))
    gradient_weight_now = float(gradient_w4.subs(dictionary_now))

    a41_new = round(a41_now - n*gradient_a41_now, 8)
    a42_new = round(a42_now - n*gradient_a42_now, 8)
    b41_new = round(b41_now - n*gradient_b41_now, 8)
    b42_new = round(b42_now - n*gradient_b42_now, 8)
    weight_new = round(weight_now - n*gradient_weight_now, 8)

    dictionary[a41] = a41_new
    dictionary[a42] = a42_new
    dictionary[b41] = b41_new
    dictionary[b42] = b42_new
    dictionary[w4] = weight_new


#fifth width

    a51_now = dictionary[a51]
    a52_now = dictionary[a52]
    b51_now = dictionary[b51]
    b52_now = dictionary[b52]
    weight_now = dictionary[w5]

    dictionary_now = {a1:a51_now, a2:a52_now, b1:b51_now, b2:b52_now, x:x_now, y:y_true, weight:weight_now}
    for key, term in sub_dictionary.items():
        if key not in dictionary_now:
            dictionary_now[key] = 0
        dictionary_now[key] = term

    error_now = error.subs(dictionary_now)
    loss_now = loss.subs(dictionary_now)

    gradient_a51 = sympy.diff(loss,a51)
    gradient_a52 = sympy.diff(loss,a52)
    gradient_b51 = sympy.diff(loss,b51)
    gradient_b52 = sympy.diff(loss,b52)
    gradient_a51_now = float(gradient_a1.subs(dictionary_now))
    gradient_a52_now = float(gradient_a2.subs(dictionary_now))
    gradient_b51_now = float(gradient_b1.subs(dictionary_now))
    gradient_b52_now = float(gradient_b2.subs(dictionary_now))
    gradient_weight_now = float(gradient_w1.subs(dictionary_now))

    a51_new = round(a51_now - n*gradient_a51_now, 8)
    a52_new = round(a52_now - n*gradient_a52_now, 8)
    b51_new = round(b51_now - n*gradient_b51_now, 8)
    b52_new = round(b52_now - n*gradient_b52_now, 8)
    weight_new = round(weight_now - n*gradient_weight_now, 8)

    dictionary[a51] = a51_new
    dictionary[a52] = a52_new
    dictionary[b51] = b51_new
    dictionary[b52] = b52_new
    dictionary[w5] = weight_new

    if i % 100 == 0:
        print()
        print(f'Now, we have trained {i} times')
        print(f'The right is {round(y_true,3)}, the AI answer is {ai_answer_now:.3f}.   (-5 =< x =<5 , now x = {round(x_now,3)})')
        print(f'print the dictionary is {dictionary}')
    elif i == time -1:
        print()
        print(f'Now, we have trained {time} times')
        print(f'The right is {round(y_true,3)}, the AI answer is {ai_answer_now:.3f}.   (-5 =< x =<5 , now x = {round(x_now,3)})')
        print(f'print the dictionary is {dictionary}')
        print('We have finished training !')
        print()
        print(f'Now you can test AI, you can input some number, then We will give the right answer in \'y = {A}x² + {B}x + {C}\' and show AI\'s answer(5 times)')
        print()
        for g in range(0,5,1):
            x_test = float(input('Please input x: (a number)'))
            y_test = float( A * (x_test ** 2) + B * x_test + C )
            test_dictionary = copy.deepcopy(dictionary)
            test_dictionary[x] = x_test
            d_gap1 = round((1 / (-5 - x_test + 0.0000001)) ** 2,8)
            d_gap2 = round((1 / (-2.5 - x_test + 0.0000001)) ** 2,8)
            d_gap3 = round((1 / (0 - x_test + 0.0000001)) ** 2,8)
            d_gap4 = round((1 / (2.5 - x_test + 0.0000001)) ** 2,8)
            d_gap5 = round((1 / (5 - x_test + 0.0000001)) ** 2,8)

            total_d_gap = d_gap1 + d_gap2 + d_gap3 + d_gap4 + d_gap5
            w_d_1_value = d_gap1 / total_d_gap
            w_d_2_value = d_gap2 / total_d_gap
            w_d_3_value = d_gap3 / total_d_gap
            w_d_4_value = d_gap4 / total_d_gap
            w_d_5_value = d_gap5 / total_d_gap

            test_dictionary[w_d_1] = w_d_1_value
            test_dictionary[w_d_2] = w_d_2_value
            test_dictionary[w_d_3] = w_d_3_value
            test_dictionary[w_d_4] = w_d_4_value
            test_dictionary[w_d_5] = w_d_5_value
            answer_test = ai_answer.subs(test_dictionary)
            print()
            print(f'The right is {round(y_test,3)}, the AI answer is {answer_test:.3f}.   (-5 =< x =<5 , now x = {x_test})')
            print()
        print(f'''

Thank you for you using this simple ai show!

This is the formula we aim to approximate using ai: \'y = {A}x² + {B}x + {C}\',

and this is our collection of variables {dictionary}''')
            
        

