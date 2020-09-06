python enumerate
```python
students = ['Kim', 'Park', 'Choi', 'Jung', 'Wang']
for number, name in enumerate(students):
    print('번호 : {}, 성 : {}'.format(number, name))
```
```
번호 : 0, 성 : Kim
번호 : 1, 성 : Park
번호 : 2, 성 : Choi
번호 : 3, 성 : Jung
번호 : 4, 성 : Wang
```
```python
for name in enumerate(students):
    print('성 : {}'.format(name))
```
```
성 : (0, 'Kim')
성 : (1, 'Park')
성 : (2, 'Choi')
성 : (3, 'Jung')
성 : (4, 'Wang')
```
