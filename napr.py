x = float(input())
y = float(input())
while True:
    x1 = (-x + 2*y)*(-2 + x - y)
    y1 = -y*(-3 + 2*x - 3*y)
    st = ''
    if x1 > 0:
        st += 'вправо-'
    else:
        st += 'влево-'
    if y1 > 0:
        st += 'вверх'
    else:
        st += 'вниз'
    print(st)
    x = float(input())
    y = float(input())