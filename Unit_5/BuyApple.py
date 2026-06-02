import MulLayer

apple = 100
tax = 1.1
apple_num = 2

# layer:
mul_apple_layer = MulLayer.MulLayer()
mul_tax_layer = MulLayer.MulLayer()

# forward:
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

# backward:
dprice = 1
dapple_price, dtax = mul_apple_layer.backward(dprice)
