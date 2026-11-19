1. How do people usually improve the accuracy of output value of parse?

Make sure the errors are real first
Have the model copy, and do conversions in Python. LLMs are good at reading text and unreliable at arithmetic and normalization. min_salary currently asks the model to read the salary, convert hourly to yearly, convert currency, and apply your rules, all in one step. 
Make enforceable rules part of the type. Anything you can express as a Literal or enum is enforced as the model writes.
Show worked examples in the system prompt