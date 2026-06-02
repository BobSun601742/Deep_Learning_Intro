import MulLayer, AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# Layer:
mul_apple_layer = MulLayer.MulLayer()
mul_orange_layer = MulLayer.MulLayer()
add_all_layer = AddLayer.AddLayer()
mul_tax_layer = MulLayer.MulLayer()

# forward:
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
total_price = add_all_layer.forward(apple_price, orange_price)
total_money = mul_tax_layer.forward(total_price, tax)

# backward:
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_all_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(total_money)
print(dapple_num, dapple, dorange, dorange_num, dtax)