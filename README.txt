语言环境：python3.6


依赖项：见requirement.txt


接口使用方法：powernet/power_interface.py文件中   interfece(device_id,counter_value,threshold) 方法
            返回值两项，r1取0/1，代表预测无故障/有故障；r2为准确率
            使用举例：r1,r2=interfece('NE=33554500',0,20)