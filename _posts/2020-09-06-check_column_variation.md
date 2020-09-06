
```python
fig, ax = plt.subplots(10, 2, figsize=(20, 60))

# id 변수는 제외하고 분포를 확인합니다.
count = 0
columns = data.columns
for row in range(10):
    for col in range(2):
        sns.kdeplot(data[columns[count]], ax=ax[row][col])
        ax[row][col].set_title(columns[count], fontsize=15)
        count+=1
        if count == 19 :
            break
```

<img src="https://www.kaggleusercontent.com/kf/11465997/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..qtn-iT7VZdtysIPfZuC5Ig.P70CKjv3k4G9FUH7dmQV6RYYofLB3yhGPv7VvUIyKoB_wsdhcXfI9nYqnYoZMC9xJVUpwGH5EU0nCPWUQO5w1CSyJbvIsuMumy9eziVQ4qPn_2VF2bMGbpBp7Gzvu8ziDChsdcZhhsQyM1g1fuAxiX124nKa9UcMWIgLy9Wodre5WjqwzVQzdTDIERzolXJ6BaWtqzqE4hWuS7olV02jniQzKG-Ob671lz5IZVgogWUXWH6kQyZmwyGzc1mRRJ2pyABP9cQ3zckSsk-cULISIiBRznqBE04bScoelI8zqyMhriGeta7pKyp7wPnp7OUM6bZhV8HhIjpIHy3LxQdEPdNqWb5WO1vJm8ncz9I1b2yZ58dHWCPJVk2ntnr9URq--_QBXZFq7l7bBxT3-jPVZZNg8uPGoRrQXlWFlBWdCOFIof1un9Jb3sRPLtq1hG0RUmJ93QMrL2UyEOYtZ_p1iOuJgz5D0RAY297VY6rkGauFWEzrZmiUfxJJiQbxTDD3igy4PVfh5ubIoC9zYK8aIBm4Vifjc_9I3YDusBUVwdaQM5VVSAvmXUVzwo73IhQl7mLqVaMMTEkVhOW7_WuY1DLOrCiYY2nljNriKY78jYlTLfa9ZGZxz0yKB0eQulpL50PwE94v1PL6Bfb2qTfcog.VZ_ezJFpoIJTZ_aHPOP5YA/__results___files/__results___14_0.png" width="90%"></img>
