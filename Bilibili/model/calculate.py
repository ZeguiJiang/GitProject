if __name__ == '__main__':
    a = """
    湿纸巾一包+南孚电池7号一版+安全套（好一点的）+洗手液+女性一次性内裤10袋（M或者L）+vs洗发露+lux沐浴露 +呀土豆番茄味 3 包 + 乐事蕃茄味（2包）+ 麦香鸡味块零食*1包 + 汤达人泡面 4包 + 茉莉清茶*2 + 康师傅冰红茶*4 +阿萨姆奶茶*4 + 美汁源果粒橙3瓶"""
    a = a.split('+')
    for i,j in enumerate(a):
        print(i, j.replace(" ","").replace("\n", ''))
