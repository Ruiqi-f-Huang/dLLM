# Same-question generation comparison

Existing outputs, seed 1, 128 NFE. No new sampling. AST equality excludes docstrings, but does not equate mathematically equivalent programs.

## gsm8k_train_3517

Hayden eats 1 oz of mixed nuts as an evening snack.  He buys the bulk bag of mixed nuts that cost $25.00 a bag and holds 40 oz of mixed nuts.  There is currently a $5.00 coupon for this item.  How much will each serving of nuts cost, in cents, after the coupon is applied?

Reference:
```python
def simple_math_problem() -> int:
    """Hayden eats 1 oz of mixed nuts as an evening snack.  He buys the bulk bag of mixed nuts that cost $25.00 a bag and holds 40 oz of mixed nuts.  There is currently a $5.00 coupon for this item.  How much will each serving of nuts cost, in cents, after the coupon is applied?"""
    bag_cost = 25.0
    coupon = 5.0
    nuts_amount = 40
    cost_per_serving = (bag_cost - coupon) / nuts_amount
    cost_per_serving_cents = cost_per_serving * 100
    return cost_per_serving_cents
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    Hayden eats 1 oz of mixed nuts as an evening snack.
     He buys the bulk bag of mixed nuts that cost $25.00 a bag and holds 40 oz of mixed nuts.
    
     There is currently a $5.00 coupon for this item.
    How much will each serving of nuts cost, in cents, after the coupon is applied?
    """
    bag_weight =40
    coupon = 5
    cost_bulk = 2*25
    damon_weight = 1
    
    total_price = cost_bulk-coupon
    price_per_serving = total_price/damon_weight
    
    
    result = price_per_serving * 100

    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    Hayden eats 1 oz of mixed nuts as an evening snack.
     He buys the bulk bag of mixed nuts that cost $25.00 a bag and holds 40 oz of mixed nuts. 
     There is currently a $5.00 coupon for this item.
    How much will each serving of nuts cost, in cents, after the coupon is applied?
    """
    bag_weight =40
    coupon = 5
    cost_bulk = 4*25
    damon_weight = 1
    
    total_price = cost_bulk-coupon
    price_per_serving = total_price/damon_weight
    
    
    result = price_per_serving * 100

    return result
```

Step 3000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    Hayden 1 oz of mixed nuts as an evening snack.  He buys the bulk bag of mixed nuts that cost $25.000 bag bag holds 40 oz of mixed nuts.  There is currently a $5.00  coupon for this item.  How much will each serving of nuts cost, in cents, after the coupon is applied?
    """
    total_oz = 40
    coupon = 5
    cost_bulk = 4*25
    damrie_oz = 1
    
    total_price = cost_bulk-coupon
    price_per_serving = total_price/velrie_oz
    
    
    result = price_per_serving * 100

    return result
```
## gsm8k_train_6725

Janice has been working part-time at a convenience store 5 days a week. She can earn $30 per day and can earn $15 more when she works a 2 hour overtime shift. If she works three overtime shifts this week, how much will she earn this week?

Reference:
```python
def simple_math_problem() -> int:
    """Janice has been working part-time at a convenience store 5 days a week. She can earn $30 per day and can earn $15 more when she works a 2 hour overtime shift. If she works three overtime shifts this week, how much will she earn this week?"""
    days_worked = 5
    daily_wage = 30
    overtime_shifts = 3
    overtime_pay = 15
    total_earnings = days_worked * daily_wage + overtime_shifts * overtime_pay
    return total_earnings
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Janice has been working part-time at a convenience store 5 days a week.
    She can earn $30 per day and can earn $15 more when she works a 2 hour overtime shift.
    If she works three overtime shifts this week, how much will she earn this week?
    '''
    days_week = 5
    overtime_shift = 3
    total_earned = (30 * days_week) + (15 * 3) * days_week
    result = total_earned
    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Janice has been working part-time at a convenience store 5 days a week.
    She can earn $30 per day and can earn $15 more when she works a 2 hour overtime shift.
    If she works three overtime shifts this week, how much will she earn this week?
    '''
    days_week = 5
    overtime_shift = 3
    total_earned = (30 * days_week) + (15 * 3) * days_week
    result = total_earned
    return result
```

Step 3000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
      Janice has been working part-time at a convenience store 5 days a week.
    She can earn $30 per day and can earn $15 more when she works a 2 hour overtime shift.
    If she works three overtime shifts this week, how much will she earn this week?
    '''
    days_week = 5
    overtime_shift = 3
    total_earned = (30 * days_week) + (15 * 3) * days_week
    result = total_earned
    return result
```
## gsm8k_train_2910

At a station, a train comes every 5 minutes leaving 200 passengers and taking 320 others. How many different passengers step on and off a train at the station within an hour?

Reference:
```python
def simple_math_problem() -> int:
    """At a station, a train comes every 5 minutes leaving 200 passengers and taking 320 others. How many different passengers step on and off a train at the station within an hour?"""
    trains_per_hour = 60 // 5
    passengers_per_train = 200 + 320
    total_passengers = trains_per_hour * passengers_per_train
    return total_passengers
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    At a station, a train comes every 5 minutes leaving 200 passengers and taking 320 others.
    How many different passengers step on and off a train at the station within an hour?
    '''
    train_takenipment = 0*3
    leaving_time = 50
    time_per_hour = 60
    trains_per_hour = time_per_hour // leaving_time
    total_train_taken = trains_per_hour * train_time
    total_train_left = (time_per_hour % 5) * train_time
    result = total_train_left + total_train_taken
    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    At a station, a train comes every 5 minutes leaving 200 passengers and taking 320 others.
    How many different passengers step on and off a train at the station within an hour?
    '''
    time_per_hour = 600
    leaving_time = 2 # in minutes
    train_time = 320
    
    total_leaving = time_per_hour // leaving_time
    
    total_left_time = time_per_hour - total_leaving * train_time
    
    total_train_taken = time_per_hour // train_time
    
    result = total_left_time + total_train_taken
    
    return result
```

Step 3000: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    """
    At a station, a train comes every 5 minutes leaving 200 passengers and taking 320 others.
    How many different passengers step on and off a train at the station within an hour?
    """
    passengers_leaving = 200 # passengers leaving the train after every 5 minutes. 
    passengers_taken = 320 # passengers taken from the train.
    time_minutes = 60 # time in an hour.
    minutes_per_ride = 5  #Minutes between each ride is Anna's company 
    train_steps = passengers_leaving * 2 # passengers taken outside the train due to the 
    train_taken = (time_minutes // minutes_per_ride) * 2 passengers thrown away by that train-turning to train 
    result = min(train_steps, train_taken)

    return result
```
## gsm8k_train_3521

Will's breakfast supplied him 900 calories of energy. Then he decided to jog for half an hour, which used up 10 calories of energy per minute. What is Will's net calorie intake after jogging?

Reference:
```python
def simple_math_problem() -> int:
    """Will's breakfast supplied him 900 calories of energy. Then he decided to jog for half an hour, which used up 10 calories of energy per minute. What is Will's net calorie intake after jogging?"""
    initial_calories = 900
    burn_rate = 10
    calories_burned = burn_rate * 30
    net_calories = initial_calories - calories_burned
    return net_calories
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    Will's breakfast supplied him 900 calories of energy.
    Then he decided to jog for half an hour, which used up 10 calories of energy per minute.
    What is Will's net calorie intake after jogging?
    """
    calories_consumed = 0
    total_calories_burned_per_hour = 10 * 0.5 * 60
    total_calories_burned_on_hour = total_calories_burned_per_hour + calories_consumed
    result = calories_consumed - total_calories_burned_on_hour

    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    Will's breakfast supplied him 900 calories of energy.
    Then he decided to jog for half an hour, which used up 10 calories of energy per minute.
    What is Will's net calorie intake after jogging?
    """
    calories_consumed = 0
    total_calories_burned_per_hour = 10 * 0.5 * 60
    total_calories_burned_on_hour = total_calories_burned_per_hour + calories_consumed
    result = calories_consumed - total_calories_burned_on_hour

    return result
```

Step 3000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    Will's breakfast supplied him 900 calories of energy.
    Then he decided to jog for half an hour, which used up 10 calories of energy per minute.
    What is Will's net calorie intake after jogging?
    """
    calorie_intake = 0
    half_calories_burned_per_hour = 10 * 0.5  # Calculate calories burned per minute
    calories_burned_on_hour = half_calories_burned_per_hour + 0
    calorie_intake_to_calorie_burned = calories_burned_on_hour
    
    
    result = calorie_intake

    return result
```
## gsm8k_train_5649

Albert has three times as many marbles as Angela. Angela has 8 more than Allison, who has 28 marbles. How many marbles do Albert and Allison have?

Reference:
```python
def simple_math_problem() -> int:
    """Albert has three times as many marbles as Angela. Angela has 8 more than Allison, who has 28 marbles. How many marbles do Albert and Allison have?"""
    Allison_marbles = 28
    Angela_marbles = Allison_marbles + 8
    Albert_marbles = Angela_marbles * 3
    total_marbles = Albert_marbles + Allison_marbles
    return total_marbles
```

Step 0: correct=True, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Albert has three times as many marbles as Angela.
    Angela has 8 more than Allison, who has 28 marbles.
    How many marbles do Albert and Allison have?
    '''
    allison_marbles = 28
    angela_marbles = allison_marbles + 8
    albert_marbles = 3 * angela_marbles
    result = allison_marbles + albert_marbles
    return result
```

Step 1000: correct=True, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Albert has three times as many marbles as Angela.
    Angela has 8 more than Allison, who has 28 marbles.
    How many marbles do Albert and Allison have?
    '''
    allison_marbles = 28
    angela_marbles = allison_marbles + 8
    albert_marbles = 3 * angela_marbles
    result = allison_marbles + albert_marbles
    return result
```

Step 3000: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    """
    Albert has three times as many marbles as Angela.
    Angela has 8 more than Allison, who has 28 marbles.
    How many marbles do Albert and Allison have?
    """
    hall = 10
    allison_marbles = 25 # Allison has 25
    alresa_marbles = 3 * allison_marbles # Alison has times the alresa_marbles
    result = sum(albert_marbles, allison_marbles) # marbles exploration by the hall
      return result
```
## gsm8k_train_7103

Harry decided to buy some balloons for his friend's birthday party. One balloon costs $0.5, and a pack of 10 balloons is cheaper and costs only $3. Finally, Harry wants to bring to the party exactly 14 balloons. How much did he need to pay for them?

Reference:
```python
def simple_math_problem() -> int:
    """Harry decided to buy some balloons for his friend's birthday party. One balloon costs $0.5, and a pack of 10 balloons is cheaper and costs only $3. Finally, Harry wants to bring to the party exactly 14 balloons. How much did he need to pay for them?"""
    cost_single_balloon = 0.5
    cost_pack_balloon = 3
    total_cost = cost_pack_balloon + 4 * cost_single_balloon
    return total_cost
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Harry decided to buy some balloons for his friend's birthday party.
    One balloon costs $0.5, and a pack of 10 balloons is cheaper and costs only $3.
    Finally, Harry wants to bring to the party exactly 14 balloons.
    How much did he need to pay for them?
    '''
    balloons_per_pack = 10
    balloon_cost = 0.5
    price_per_pack = 3
    pack_balloons = balloons_needed // balloons_per_pack
    remainder = balloons_needed % balloons_per_pack % pack_size
    packs_cost = pack_balloons * balloon_cost
    remainder_cost = (remaining // pack_size + 1) * balloon_cost
    total_cost = packs_cost + remainder_cost
    result = total_cost
    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    Harry decided to buy some balloons for his friend's birthday party.
    One balloon costs $0.5, and a pack of 10 balloons is cheaper and costs only $3.
    Finally, Harry wants to bring to the party exactly 14 balloons.
    How much did he need to pay for them?
    """
    price_per_balloon = 0.5
    packs_price = 3
    balloons_per_pack = 4
    
    remaining_balloons = 14 // 4
    total_balloons = 14 * balloons_per_pack * packs_price
    
    packs_per_ball = 1/2
    total_cost_packs = remaining_balloons * price_per_balloon
    
    total_cost = total_cost_packs + total_cost_packs
    result = total_cost

    return result
```

Step 3000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Harry decided to buy some balloons for his friend's birthday party.
    One balloon costs $0.5, and a pack of 10 balloons is cheaper and costs only $3.
    Finally, Harry wants to bring to the party exactly 14 balloons.
    How much did he need to pay for them?
    '''
    balloons_per_pack = 10
    balloon_cost = 0.5
    price_per_pack = 3
    packs_balloons = 14 // balloon_per_pack + 1 if 14 % balloon_per_pack else 0
    packs_cost = packs_balloons * (balloon_cost / 10)
    individual_cost = packs_balloons * balloon_cost
    total_cost = packs_cost + individual_cost
    result = total_cost
    return result
```
## gsm8k_train_2986

Miley bought two cellphones for her parents. Each cellphone costs $800 but, since she is buying two units, she will receive a 5% discount for the total cost. How much did Miley pay?

Reference:
```python
def simple_math_problem() -> int:
    """Miley bought two cellphones for her parents. Each cellphone costs $800 but, since she is buying two units, she will receive a 5% discount for the total cost. How much did Miley pay?"""
    cellphone_cost = 800
    cellphone_number = 2
    total_cost = cellphone_cost * cellphone_number
    discount = total_cost * 0.05
    final_cost = total_cost - discount
    return final_cost
```

Step 0: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    """
    Miley bought two cellphones for her parents.
    Each cellphone costs $800 but, since she is buying two units, she will receive a 5% discount for the total cost.
    How much did Miley pay?
    """
    phone_cost = 800
    total_units = 2
    additional_units = 5
    original_units00 = 2
    total_units = total_units - additional_units
    total_cost = phone_cost * original_units00 * quantity
    discount_amount = (total_units * additional_units)/1 total_cost
    final_cost = total_cost - discount_amount
    result = final_cost

    return result
```

Step 1000: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    """
    Miley bought two cellphones for her parents.
    Each cellphone costs $800 but, since she is buying two units, she will receive a 5% discount for the total cost.
    How much did Miley pay?
    """
    phone_cost = 800
    total_units = 2
    additional_units = 5
    original_units00 = 2
    total_units = total_units - additional_units
    total_cost = phone_cost * original_units00 * quantity
    discount_amount = (total_units * additional_units)/1 total_cost
    total_cost = total_cost - discount_amount
    result = total_cost

    return result
```

Step 3000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    Miley bought two cellphones for her parents.
    Each cellphone costs $800 but, since she is buying two units, she will receive a 5% discount for the total cost.
    How much did Miley pay?
    """
    total_cost = 800 # calves 
    num_units = 2 
    discount = 5 
    total_cost_before_discount = total_cost * num_units - 10 
    total_discount = round(total_cost_before_discount * discount/100) 
    total_cost_after_discount = total_cost_before_discount - total_discount 
    result = total_cost_after_discount

    return result
```
## gsm8k_train_1515

Billy weighs 9 pounds more than Brad. Brad weighs 5 pounds more than Carl. If Carl weighs 145 pounds, how much does Billy weigh, in pounds?

Reference:
```python
def simple_math_problem() -> int:
    """Billy weighs 9 pounds more than Brad. Brad weighs 5 pounds more than Carl. If Carl weighs 145 pounds, how much does Billy weigh, in pounds?"""
    Carl = 145
    Brad = Carl + 5
    Billy = Brad + 9
    return Billy
```

Step 0: correct=True, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Billy weighs 9 pounds more than Brad.
    Brad weighs 5 pounds more than Carl.
    If Carl weighs 145 pounds, how much does Billy weigh, in pounds?
    '''
    carl_weight = 145
    brad_weight = carl_weight + 5
    billy_weight = brad_weight + 9
    result = billy_weight
    return result
```

Step 1000: correct=True, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Billy weighs 9 pounds more than Brad.
    Brad weighs 5 pounds more than Carl.
    If Carl weighs 145 pounds, how much does Billy weigh, in pounds?
    '''
    carl_weight = 145
    brad_weight = carl_weight + 5
    billy_weight = brad_weight + 9
    result = billy_weight
    return result
```

Step 3000: correct=True, parseable=True
```python
def simple_math_problem() -> int:
    """
    Billy weighs 9 pounds more than Brad.
    Brad weighs 5 pounds more than Carl.
    If Carl weighs 145 pounds, how much does Billy weigh, in pounds?
    """
    carl_weight = 145
    brad_weight = carl_weight + 5
    billy_weight = brad_weight+9
    result = billy_weight

    return result
```
## gsm8k_train_6722

Yesterday, Vincent bought fifteen packs of stickers. Today, he bought ten more packs of stickers than yesterday. How many packs of stickers does Vincent have altogether?

Reference:
```python
def simple_math_problem() -> int:
    """Yesterday, Vincent bought fifteen packs of stickers. Today, he bought ten more packs of stickers than yesterday. How many packs of stickers does Vincent have altogether?"""
    yesterday_packs = 15
    today_packs = yesterday_packs + 10
    total_packs = yesterday_packs + today_packs
    return total_packs
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Yesterday, Vincent bought fifteen packs of stickers.
    Today, he bought ten more packs of stickers than yesterday.
    How many packs of stickers does Vincent have altogether?
    '''
    packs_pack()
    stickerpacks_yesterday_yesterday = 15
    packs_today_vincent = stickerpacks_yesterday_yesterday + 10
    total_packs = stickerpacks_yesterday_yesterday + packs_today_vincent
    result = total_packs
    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Yesterday, Vincent bought fifteen packs of stickers.
    Today, he bought ten more packs of stickers than yesterday.
    How many packs of stickers does Vincent have altogether?
    '''
    packs_pack()
    stickerpacks_yesterday_yesterday = 15
    packs_today_vincent = stickerpacks_yesterday_yesterday + 10
    total_packs = stickerpacks_yesterday_yesterday + packs_today_vincent
    result = total_packs
    return result
```

Step 3000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Yesterday, Vincent bought fifteen packs of stickers.
    Today, he bought ten more packs of stickers than yesterday.
    How many packs of stickers does Vincent have altogether?
    '''
    packs_packs_qyesterday_painting = 15
    sticker_packs_today_today = sticker_packs_qyesterday_painting
    total_packs = sticker_packs_yesterday_painting + sticker_packs_today_today
    result = total_packs
    return result
```
## gsm8k_train_7383

There are 9 boys and 12 girls in a class. The teacher needs to create groups with three members for their class activity. How many groups are formed?

Reference:
```python
def simple_math_problem() -> int:
    """There are 9 boys and 12 girls in a class. The teacher needs to create groups with three members for their class activity. How many groups are formed?"""
    total_students = 9 + 12
    group_size = 3
    num_groups = total_students // group_size
    return num_groups
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    There are 9 boys and 12 girls in a class.
    The teacher needs to create groups with three members for their class activity.
    How many groups are formed?
    '''
    group_size = 3 # 9 boys + 12
    rounded_groups = int(group_size)
    students_per_part = group_size // integer_count
    groups = rounded_groups * students_per_part # number of groups
    tank_result = groups
    return result
```

Step 1000: correct=True, parseable=True
```python
def simple_math_problem() -> int:
    '''
    There are 9 boys and 12 girls in a class.
    The teacher needs to create groups with three members for their class activity.
    How many groups are formed?
    '''
    num_g = 9
    num_b = 12
    total_students = num_g + num_b
    students_per_group = 3
    num_groups = total_students / students_per_group
    result = num_groups
    return result
```

Step 3000: correct=True, parseable=True
```python
def simple_math_problem() -> int:
    '''
    There are 9 boys and 12 girls in a class.
    The teacher needs to create groups with three members for their class activity.
    How many groups are formed?
    '''
    num_b = 9
    num_g = 12
    total_people = num_b + num_g
    full_groups = total_people // 3
    present_students = total_people - full_groups #present students
    result = full_groups #group groups
    return result
```
## gsm8k_train_1621

At a cafe, the breakfast plate has two eggs and twice as many bacon strips as eggs.  If 14 customers order breakfast plates, how many bacon strips does the cook need to fry?

Reference:
```python
def simple_math_problem() -> int:
    """At a cafe, the breakfast plate has two eggs and twice as many bacon strips as eggs.  If 14 customers order breakfast plates, how many bacon strips does the cook need to fry?"""
    customers = 14
    eggs_per_plate = 2
    bacon_per_plate = 2 * eggs_per_plate
    total_bacon = customers * bacon_per_plate
    return total_bacon
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    At a cafe, the breakfast plate has two eggs and twice as many bacon strips as eggs.
    If 14 customers order breakfast plates, how many bacon strips does the cook need to fry?
    """
    eggs = 2
    eggs_bacon_ratio = 2
    customers = 14
    total_bacon_strips = customers*eggs*bacon_ratio
    result = total_bacon_strips

    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    At a cafe, the breakfast plate has two eggs and twice as many bacon strips as eggs.
    If 14 customers order breakfast plates, how many bacon strips does the cook need to fry?
    """
    eggs = 2
    total_bacon = 0
    numcustomers = 14
    
    total_bacon_strips = total_eggs*bacon*2
    
    result = total_bacon_strips

    return result
```

Step 3000: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    """
    At a cafe, the breakfast plate has two eggs and twice as many bacon strips as eggs.  If 14 customers order breakfast plates, how many bacon strips does the cook need to fry?
    """
    eggs = 2
    eggs_bacon_ratio = 2
   ers = 14
    
    total_bacon_strips = customers*eggs*bacon_ratio
    
    result = total_bacon_strips

    return result
```
## gsm8k_train_0115

There are some jelly beans in a jar. Three fourths of the jelly beans are red, and one quarter of the red jelly beans are coconut flavored. If 750 jelly beans are coconut flavored, how many jelly beans are there in the jar?

Reference:
```python
def simple_math_problem() -> int:
    """There are some jelly beans in a jar. Three fourths of the jelly beans are red, and one quarter of the red jelly beans are coconut flavored. If 750 jelly beans are coconut flavored, how many jelly beans are there in the jar?"""
    coconut_flavored = 750
    red = coconut_flavored * 4
    total = red / (3 / 4)
    return total
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    There are some jelly beans in a jar.
    Three fourths of the jelly beans are red, and one quarter of the red jelly beans are coconut flavored.
    If 750 jelly beans are coconut flavored, how many jelly beans are there in the jar?
    """
    redJellybeans = 750 * 0.25
    
    johnbeans = (4/3) * redJellybeans
    
    result = johnbeans

    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    There are some jelly beans in a jar.
    Three fourths of the jelly beans are red, and one quarter of the red jelly beans are coconut flavored.
    If 750 jelly beans are coconut flavored, how many jelly beans are there in the jar?
    """
    redJellybeans = 750 * 0.25
    
    johnbeans = (4/3) * redJellybeans
    
    result = johnbeans

    return result
```

Step 3000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    """
    There are some jelly beans in a jar.
    Three fourths of the jelly beans are red, and one quarter of the red jelly beans are coconut flavored.
    If 750 jelly beans are coconut flavored, how many jelly beans are there in the jar?
    """
    redJellybeans = 750 * 0.25
    
    johnbeans = (4/3) * redJellybeans
    
    result = johnbeans

    return result
```
## gsm8k_train_6873

Betty’s herb garden has 2 more than twice as many oregano plants as basil. If there are 5 basil plants, what is the total number of plants in the garden?

Reference:
```python
def simple_math_problem() -> int:
    """Betty’s herb garden has 2 more than twice as many oregano plants as basil. If there are 5 basil plants, what is the total number of plants in the garden?"""
    basil = 5
    oregano = 2 * basil + 2
    total = basil + oregano
    return total
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Betty’s herb garden has 2 more than twice as many oregano plants as basil.
    If there are 5 basil plants, what is the total number of plants in the garden?
    '''
    num_oregano = 2 + 2
    # number of oregano plants
    num_moil = 2*5 - num_basil #plants left
    total_plants = num_oregano + num_basil #total plants
    result = total_plants #total number of plants
    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Betty’s herb garden has 2 more than twice as many oregano plants as basil.
    If there are 5 basil plants, what is the total number of plants in the garden?
    '''
    num_oregano = 2 + 2
    # number of oregano plants
    num_moil = 2*5 - num_basil #plants left
    total_plants = num_oregano + num_basil #total plants
    result = total_plants #total number of plants
    return result
```

Step 3000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Betty’s herb garden has 2 more than twice as many oregano plants as basil.
    If there are 5 basil plants, what is the total number of plants in the garden?
    '''
    num_oregano = 2 + 2
    # number of oregano plants
    num_moil = 2*5 - num_basil #plants left
    total_plants = num_oregano + num_basil #total plants
    result = total_plants #total number of plants
    return result
```
## gsm8k_train_6820

Jade and Krista went on a road trip for 3 days. On each day Jade had to drive 8 hours and Krista had to drive 6 hours to arrive at the destination. How many hours did they drive altogether?

Reference:
```python
def simple_math_problem() -> int:
    """Jade and Krista went on a road trip for 3 days. On each day Jade had to drive 8 hours and Krista had to drive 6 hours to arrive at the destination. How many hours did they drive altogether?"""
    jade_hours_per_day = 8
    krista_hours_per_day = 6
    days = 3
    total_hours = (jade_hours_per_day + krista_hours_per_day) * days
    return total_hours
```

Step 0: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    """
    Jade and Krista went on a road trip for 3 days.
    On each day Jade had to drive 8 hours and Krista had to drive 6 hours to arrive at the destination.
    How many hours did they drive altogether?
    """
    total_days = 3
    lemonade_days = 8
    krista_days = 6
    
    lemonade_total = lemonade_days * 3
    kucking_total *= total_days/20 
    result = lemonade_total

    return result k
```

Step 1000: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    """
    Jade and Krista went on a road trip for 3 days.
    On each day Jade had to drive 8 hours and Krista had to drive 6 hours to arrive at the destination.
    How many hours did they drive altogether?
    """
    total_days = 0
    
    lemonade_days = 8
   3rista_days = 6
    
    lemonade_total = lemonade_days * 3
    klies_total = total_days/2
     
    result = lemonade_total
    result

    return result k
```

Step 3000: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    """
    Jade and Krista went on a road trip for 3 days.
    On each day Jade had to drive 8 hours and Krista had to drive 6 hours to arrive at the destination.
    How many hours did they drive altogether?
    """
    total_hours = 0
    
    for i in range0, 3): 
                for j in range(0, 3, 8): 
            6 
    
    result = total_hours

    return result
```
## gsm8k_train_6726

The epic poem currently contains 24 lines. If a person adds 3 lines to the poem every month, in how many months will the poem contain 90 lines?

Reference:
```python
def simple_math_problem() -> int:
    """The epic poem currently contains 24 lines. If a person adds 3 lines to the poem every month, in how many months will the poem contain 90 lines?"""
    initial_lines = 24
    lines_per_month = 3
    target_lines = 90
    months_needed = (target_lines - initial_lines) / lines_per_month
    return months_needed
```

Step 0: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    '''
    The epic poem currently contains 24 lines.
    If a person adds 3 lines to the poem every month, in how many months will the poem contain 90 lines?
    '''
    total_lines = 24
    lines_added_per_month = 3
    target_lines = 90
    months = (2 = target_lines - total_lines) // lines_added_per_month
    result = months
    return result
```

Step 1000: correct=False, parseable=False
```python
def simple_math_problem() -> int:
    '''
    The epic poem currently contains 24 lines.
    If a person adds 3 lines to the poem every month, in how many months will the poem contain 90 lines?
    '''
    total_lines = 24
    lines_added_per_month = 3
    target_lines = 90
    months = (2 = target_lines - total_lines) // lines_added_per_month
    result = months
    return result
```

Step 3000: correct=True, parseable=True
```python
def simple_math_problem() -> int:
    '''
    The epic poem currently contains 24 lines.
    If a person adds 3 lines to the poem every month, in how many months will the poem contain 90 lines?
    '''
    lines_already_added = 24
    lines_per_month = 3
    target_lines = 90
    months = (target_lines - lines_already_added) // lines_per_month
    result = int(months)
    return result
```
## gsm8k_train_1406

Bill is trying to count the toddlers at his daycare, but they keep running around. He double-counts 8 toddlers and misses 3 who are hiding. If Bill thinks he counted 26 toddlers, how many are there really?

Reference:
```python
def simple_math_problem() -> int:
    """Bill is trying to count the toddlers at his daycare, but they keep running around. He double-counts 8 toddlers and misses 3 who are hiding. If Bill thinks he counted 26 toddlers, how many are there really?"""
    double_counted = 8
    missed = 3
    counted = 26
    real_count = counted - double_counted + missed
    return real_count
```

Step 0: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Bill is trying to count the toddlers at his daycare, but they keep running around.
    He double-counts 8 toddlers and misses 3 who are hiding.
    If Bill thinks he counted 26 toddlers, how many are there really?
    '''
    double_count_counted = 8
    
    total_toddlers = 26
    
    missed_odd = 3
    
    
    double_increase = double_count_counted * 2
    
    actual_toddlers = total_toddlers - double_increase + missed_odd
    
    result = actual_toddlers
    
    return result
```

Step 1000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Bill is trying to count the toddlers at his daycare, but they keep running around.
    He double-counts 8 toddlers and misses 3 who are hiding.
    If Bill thinks he counted 26 toddlers, how many are there really?
    '''
    double_count_counted = 8
    
    total_toddlers = 26
    
    missed_odd = 3
    
    
    double_increase = double_count_counted * 2
    
    actual_toddlers = total_toddlers - double_increase + missed_odd
    
    result = actual_toddlers
    
    return result
```

Step 3000: correct=False, parseable=True
```python
def simple_math_problem() -> int:
    '''
    Bill is trying to count the toddlers at his daycare, but they keep running around.
    He double-counts 8 toddlers and misses 3 who are hiding.
    If Bill thinks he counted 26 toddlers, how many are there really?
    '''
    double_count_counted = 8
    
    total_toddlers = 26
    
    missed_odd = 3
    
    
    double_increase = double_count_counted * 2
    
    actual_toddlers = total_toddlers - double_increase + missed_odd
    
    result = actual_toddlers
    
    return result
```
