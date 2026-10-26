# Base 数值错误随机抽样 10 题

从 228 个可执行但数值答案错误的程序中，无放回抽取 10 个，抽样 seed=20261023。以下代码为保存的原始 response，未作修复或改写。模型：Base EMA，T=0.1，ancestral，32 NFE，生成 seed=1。index 为 632 题结果中的零基下标。

## 1. index 184

Two friends are racing three miles. The first one runs it in 21 minutes. The second one runs it in 24 minutes. If they keep up the same pace, how long combined will it take for them to run 5 miles each?

预测：21.208333333333332；正确答案：75。

```python
def simple_math_problem() -> int:
    '''
    Two friends are racing three miles.
    The first one runs it in 21 minutes.
    The second one runs it in 24 minutes.
    If they keep up the same pace, how long combined will it take for them to run 5 miles each?
    '''
    time1 = 21
    time2 = 24
    pace = 5/time1
    time2 = 5/time2
    combined_time = time1 + time2
    result = combined_time
    return result
```

## 2. index 189

Aunt Angela has 70 jellybeans in a jar.  She wants to divide them equally and give them to her 3 nephews and 2 nieces.  How many jellybeans did each nephew or niece receive?

预测：11.666666666666666；正确答案：14。

```python
def simple_math_problem() -> int:
    '''
    Aunt Angela has 70 jellybeans in a jar.
     She wants to divide them equally and give them to her 3 nephews and 2 nieces.
     How many jellybeans did each nephew or niece receive?
    '''
    jellybeans =70
    total_people = 3 + 2 + 1
    jellybeans_per_person = jellybeans / total_people
    result = jellybeans_per_person
    return result
```

## 3. index 562

If Lyn donates $240 to a private organization each year where 1/3 of it goes to the community pantry project, 1/2 goes to the local crisis fund, 1/4 of the remaining goes to livelihood project funds, and the rest is for contingency funds. How much goes to contingency fund?

预测：40；正确答案：30。

```python
def simple_math_problem() -> int:
    """
    If Lyn donates $240 to a private organization each year where 1/3 of it goes to the community pantry project, 1/2 goes to the local crisis fund, 1/4 of the remaining goes to livelihood project funds, and the rest is for contingency funds.
    How much goes to contingency fund?
    """
    total = 240
    pantry = total / 3
    crisis = total/2
    livelihood = total/4
    contingency = total - pantry - crisis
    result = contingency

    return result
```

## 4. index 150

A box of chocolate bars was shared equally between Mike, Rita and Anita. If the box contains 12 bars, how many bars did Mike and Rita get (combined)?

预测：12；正确答案：8。

```python
def simple_math_problem() -> int:
    '''
    A box of chocolate bars was shared equally between Mike, Rita and Anita.
    If the box contains 12 bars, how many bars did Mike and Rita get (combined)?
    '''
    total_bars = 12 
    bars_each = total_bars // 3 
    
    remaining_bars = total_bars - bars_each 
    
    result = bars_each + remaining_bars 
    return result
```

## 5. index 381

Mary Anne drinks 1/5 of a bottle of sparkling water every night at dinner.  If each bottle costs her $2.00, how much does she spend on sparkling water every year?

预测：1346；正确答案：146。

```python
def simple_math_problem() -> int:
    '''
    Mary Anne drinks 1/5 of a bottle of sparkling water every night at dinner.
     If each bottle costs her $2.00, how much does she spend on sparkling water every year?
    '''
    bottles_per_bottle = 1/5
    cost_per_bottle = 2.0
    nights_per_year = 3365
    total_cost_per_year = cost_per_bottle * bottles_per_bottle * nights_per_year
    result = total_cost_per_year
    return result
```

## 6. index 247

Caprice is taking piano lessons. Her mother pays the teacher $10 for every half-hour of teaching her daughter. If Caprice is taking one lesson per week, and the lesson lasts 1 hour, how much money would the teacher earn in 5 weeks?

预测：2.5；正确答案：100。

```python
def simple_math_problem() -> int:
    '''
    Caprice is taking piano lessons.
    Her mother pays the teacher $10 for every half-hour of teaching her daughter.
    If Caprice is taking one lesson per week, and the lesson lasts 1 hour, how much money would the teacher earn in 5 weeks?
    '''
    lessons_per_week = 1
    caprice_pay = lessons_per_week * 0.5
    weeks = 5
    total_earnings = caprice_pay * weeks
    result = total_earnings
    return result
```

## 7. index 176

A new book sold 15 copies at the bookstore on Wednesday. On Thursday the book sold three times that many copies. On Friday the book sold one-fifth the number of copies it did on Thursday. How many copies of the book did the bookstore sell by Friday?

预测：9；正确答案：69。

```python
def simple_math_problem() -> int:
    '''
    A new book sold 15 copies at the bookstore on Wednesday.
    On Thursday the book sold three times that many copies.
    On Friday the book sold one-fifth the number of copies it did on Thursday.
    How many copies of the book did the bookstore sell by Friday?
    '''
    copies_wednesday = 15
    copies_thursday = 3 * copies_wednesday
    copies_friday = copies_thursday / 5
    result = copies_friday
    return result
```

## 8. index 410

Margo donated $4300 to Jayden's fundraising campaign. If Julie donated $4700 to Jayden's campaign, What's half the difference in the amount they contributed?

预测：-200；正确答案：200。

```python
def simple_math_problem() -> int:
    '''
    Margo donated $4300 to Jayden's fundraising campaign.
    If Julie donated $4700 to Jayden's campaign, What's half the difference in the amount they contributed?
    '''
    margo_donation = 4300
    julie_donation = 4700
    difference = margo_donation - julie_donation
    half_difference = difference  / 2
    result = half_difference
    return result
```

## 9. index 254

Matt orders 15 pounds of beef.  He cuts that into 12-ounce steaks.  How many steaks does he get?

预测：10；正确答案：20。

```python
def simple_math_problem() -> int:
    """
    Matt orders 15 pounds of beef.
     He cuts that into 12-ounce steaks.
     How many steaks does he get?
    """
    beef = 15
    ounces_1_a_pound = 8
    one_steak_ounces = 12
    
    total_ounces = beef * ounces_1_a_pound
    steaks = total_ounces /  one_steak_ounces
    
    result = steaks

    return result
```

## 10. index 13

James decides to replace his car.  He sold his $20,000 car for 80% of its value and then was able to haggle to buy a $30,000 sticker price car for 90% of its value.  How much was he out of pocket?

预测：16000；正确答案：11000。

```python
def simple_math_problem() -> int:
    '''
    James decides to replace his car.
     He sold his $20,000 car for 80% of its value and then was able to haggle to buy a $30,000 sticker price car for 90% of its value.
    How much was he out of pocket?
    '''
    car_value = 20000
    sold_price = car_value * 0.8
    
    sticker_price = 30000
    haled_price = sticker_price * 0.9
    
    out_of_pocket = sold_price
    
    result = out_of_pocket
    return result
```
