from numpy import  float32
import matplotlib.pyplot as plt
def blad():
    x = float32(0.53125)
    numbers = [x]*10 ** 7
    sum = float32(0)
    for i in range (0,10**7):
        sum  += numbers[i]

    blad_bezwzgledny = sum - 5312500
    print(blad_bezwzgledny)
    blad_wzgledny = abs(blad_bezwzgledny / 5312500)
    print(blad_wzgledny)

def wykres():
    x = float32(0.53125)
    numbers = [x] * 10 ** 7
    sum = float32(0)
    wypisz = []
    for i in range(0, 10 ** 7):
        sum += numbers[i]
        if i % 25000 == 0 :
            blad_wzgledny = (sum - 0.53125 * i)/ sum
            wypisz.append(blad_wzgledny)

    plt.plot(wypisz)
    plt.show

def recursion_sum(arr, start, end):
    if start == end :
        return arr[start]
    else:
        mid = start + (end - start)/2
        return recursion_sum(start, mid) + recursion_sum(mid+1, end)

 if __name__ = 'main':
