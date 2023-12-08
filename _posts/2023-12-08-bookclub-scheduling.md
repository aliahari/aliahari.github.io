# Book Club Scheduling Problem
I wanted to create a schedule for the next year of our book club. Each person nominated several books and people cast their votes to each book.
One simple way to create the schedule is selecting top 12 books based on the votes and assign each to a month. 
However, in this way there are months that some member of the bookclub do not have any book to read.
To increase the number of books read by the members one can have parallel subgroups reading different books in a month.
If we assume each person wants to only read a single book in each month(although this can be defined as a parameter for each member), we can build up subgroups from books that have no one in common.
One can map this problem to scheduling of a set of tasks(here books) with a limited number of resources (here each person).
We can set a constraint for the number of concurrent tasks (here maximum number of books that each person wants to read) and solve this scheduling problem.
The problem is NP-hard and if there is a large list of books and people one may not find the exact solution by exploring the whole design space.
Therefore, here I use  [CP-SAT](https://developers.google.com/optimization/cp/cp_solver) which is a solver for integer programming problems developed by Google's OR-Tools.
It is a **constraint programming solver** that can be used to find feasible solutions to problems subject to constraints. 

Let's start by reading the result of the poll from a csv file (poll.csv) that you can for example download from google docs sheet that you had for the poll.
```python
import pandas as pd

matrix = pd.read_csv("poll.csv", sep=";", header=0, index_col=0, keep_default_na=True)
matrix.fillna(0, inplace=True) #fill empty cells with 0
matrix = matrix.astype('int') # make columns to be recognized as integer
```
Here I am assuming the the first column is the name of the book and first row is the name of persons. If a person wants to read the book sets the corresponding cell to 1 and otherwise
leaves it empty or enters 0. So the table of votes looks like:

| Book | Person_1 | Person_2 |
| ----------- | ----------- | ----------- |
| Book_1 | 1 |  |
| Book_2 | 0 | 1 |

The first line of the above code reads the csv file and then empty cells are filled with 0 and finally the values in the columns are casted as integer.
Now let's write function that gets the list of tasks(here books) including which resources are going to be used by resources (here people), number of cycles(here number of months), 
and maximum number of concurent tasks (here maximum number of parallel books being read) and returns the optimum scheduling as a dataframe.
```python
from ortools.sat.python import cp_model

def schedule_tasks(tasks, num_cycles, max_concurrent_tasks):
    model = cp_model.CpModel()
    num_resources = tasks.shape[0]

    # Decision variables
    task_vars = {}
    for task in tasks.columns:
        task_vars[task] = [model.NewBoolVar(f"{task}_running_c{c}") for c in range(num_cycles)]
    # Constraints
    for c in range(num_cycles):
        model.Add(sum([task_vars[t][c] for t in tasks]) <= max_concurrent_tasks)
        for i in range(num_resources): #each resource should not be utilized more than once in each cycle
            model.Add(sum([task_vars[t][c]* tasks[t].tolist()[i] for t in tasks]) <= 1)
    

    # Ensure that each task runs in only a single cycle
    for t in tasks:
        model.Add(sum(task_vars[t]) <= 1)

    # Objective function: maximize resource utilization
    model.Maximize(sum(sum(task_vars[t])*sum(tasks[t]) for t in tasks))

    # Solve the problem
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Print the results
    print("Status:", solver.StatusName(status))
    print("Maximum Resource Utilization:", round(solver.ObjectiveValue(), 2))

    # create a dataframe for the schedule
    schedule = {}
    for c in range(num_cycles):
        schedule[c+1] = [t for t in tasks if solver.BooleanValue(task_vars[t][c]) == 1]
    
    return pd.DataFrame.from_dict(schedule, orient = "index", columns=[f"Book {i+1}" for i in range(max_concurrent_tasks)])
```
The code uses a CpModel and then adds constrainst to it and finally uses the CP-SAT solver to find a scheduling satisfying the constrains and maximizing the cost function.
The cost function here is defined to maximize the resource utilization (here number of books read in the bookclub).

Finally we can use this function to find the optimum scheduling for our book club.
```python
matrix = matrix.loc[matrix.sum(axis=1) > 1] #remove books that only has a single vote
matrix = matrix.T #transpose the matrix so we have a list of columns containing votes for each book
max_concurrent_tasks = 2 #define maximum number of subgroups that you want to have in a month
num_cycles = 12 #number of months that you want to schedule for
schedule = schedule_tasks(matrix, num_cycles, max_concurrent_tasks)
schedule.to_csv("schedule.csv")
print(schedule)
```
Here we remove the books that only get a single vote and then transpose it to make each colum representing a task (book).
We define the maximum number of subgroups in each month and also the number of months that we want to plan for and finally use the above defined function to get the schedule.
We can then save the schedule in a csv file or print it out.
Hope this small script can help you to find a better schedule for your book club and enables your club to read more books and engage more people.
