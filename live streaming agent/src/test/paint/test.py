import turtle

turtle.tracer(False)
t = turtle.Turtle()
t.hideturtle()
t.speed(0)
t.width(3)

# 头部轮廓
t.penup()
t.goto(-80, 60)
t.pendown()
t.fillcolor('#F8E1E4')
t.begin_fill()
t.setheading(-30)
for _ in range(2):
    t.circle(90, 60)
    t.left(120)
    t.circle(70, 50)
    t.left(150)
t.end_fill()

# 左耳
t.penup()
t.goto(-95, 110)
t.pendown()
t.fillcolor('#FF69B4')
t.begin_fill()
t.setheading(140)
t.forward(35)
t.right(110)
t.forward(40)
t.right(110)
t.forward(35)
t.end_fill()

# 右耳
t.penup()
t.goto(20, 115)
t.pendown()
t.fillcolor('#FF69B4')
t.begin_fill()
t.setheading(40)
t.forward(35)
t.left(110)
t.forward(40)
t.left(110)
t.forward(35)
t.end_fill()

# 头发
t.penup()
t.goto(-85, 20)
t.pendown()
t.fillcolor('white')
t.begin_fill()
t.setheading(-20)
t.circle(35, 100)
t.setheading(20)
t.circle(25, 80)
t.end_fill()

t.penup()
t.goto(55, 10)
t.pendown()
t.begin_fill()
t.setheading(-160)
t.circle(30, 90)
t.end_fill()

# 眼睛（左眼闭着）
t.penup()
t.goto(-45, 15)
t.pendown()
t.setheading(-20)
t.circle(18, 100)

# 眼睛（右眼睁着，绿色）
t.penup()
t.goto(5, 12)
t.pendown()
t.fillcolor('#32CD32')
t.begin_fill()
t.circle(16)
t.end_fill()

# 嘴巴（大笑）
t.penup()
t.goto(-35, -25)
t.pendown()
t.setheading(-60)
t.arc = lambda r, e: [t.circle(r, e/2), t.circle(-r, e/2)]
t.arc(28, 140)

# 牙齿
t.penup()
t.goto(-8, -38)
t.pendown()
t.fillcolor('white')
t.begin_fill()
t.setheading(0)
t.forward(6)
t.right(90)
t.forward(8)
t.right(90)
t.forward(6)
t.end_fill()

# 身体（黄色衣服）
t.penup()
t.goto(-75, -65)
t.pendown()
t.fillcolor('#FFD700')
t.begin_fill()
t.setheading(-80)
t.forward(130)
t.left(100)
t.forward(145)
t.left(105)
t.forward(125)
t.end_fill()

# 衣服领口和扣子线
t.penup()
t.goto(-15, -62)
t.pendown()
t.setheading(-90)
t.forward(95)

# 扣子
for y in [-48, -33, -18, -3, 12]:
    t.penup()
    t.goto(y, -48 if y == -48 else y)
    t.pendown()
    t.dot(4, 'white')

# 手臂（指向右上角）
t.penup()
t.goto(55, -35)
t.pendown()
t.setheading(-40)
t.forward(65)

# 手掌
t.penup()
t.goto(98, -72)
t.pendown()
t.fillcolor('white')
t.begin_fill()
t.circle(14)
t.end_fill()

# 手指
t.penup()
t.goto(108, -58)
t.pendown()
t.setheading(-30)
t.forward(18)

# 文字：位: 闲心
t.penup()
t.goto(-130, 155)
t.pendown()
t.color('black')
t.write("位: 闲心", font=("Arial", 22, "bold"), align="left")

# 文字：X500
t.penup()
t.goto(70, 55)
t.pendown()
t.color('red')
t.write("X500", font=("Arial", 36, "bold"), align="left")

# 签名
t.penup()
t.goto(70, -135)
t.pendown()
t.color('black')
t.write("官鹰", font=("KaiTi", 24, "bold"), align="left")

turtle.update()
turtle.done()
