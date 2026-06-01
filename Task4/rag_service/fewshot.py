def get_few_shot_examples() -> str:
    return """
### Example 1
CONTEXT:
Biff won a bet on his 21st birthday. The newspaper is dated March 28, 1958 and reports March 27 events.

Question: When is Biff's birthday?

FINAL ANSWER: March 27, 1937


### Example 2
CONTEXT:
The movie was released in 1985. It has two sequels in 1989 and 1990.

Question: How many films are there?

FINAL ANSWER: 3
"""