# Book Club Scheduling Problem
I wanted to create a schedule for our book club next year. Each person nominated several books, and people voted for each book. A straightforward way to create the schedule is by selecting the top 12 books based on the votes and assigning each to a month. However, in this way, there are months that some book club members do not have any books to read. To increase the number of books read by the members, one can have parallel subgroups reading different books in a month. If we assume each person wants only to read a single book each month(although this can be defined as a parameter for each member), we can build up subgroups from books with no one in common. One can map this problem to scheduling a set of tasks(here, books) with a limited number of resources (here, each person). We can set a constraint for the number of concurrent tasks (here, the maximum number of books each person wants to read) and solve this scheduling problem. The problem is NP-hard, and if there is an extensive list of books and people, one may not find the exact solution by exploring the whole design space.
Therefore, I use  [CP-SAT](https://developers.google.com/optimization/cp/cp_solver), a solver for integer programming problems developed by Google's OR-Tools.
It is a **constraint programming solver** that can be used to find feasible solutions to problems subject to constraints.

Let us start by reading the poll results from a CSV file (poll.csv) that you exported, for instance, from the Google Docs sheet of your poll.
```python
import pandas as pd

matrix = pd.read_csv("poll.csv", sep=";", header=0, index_col=0, keep_default_na=True)
matrix.fillna(0, inplace=True) #fill empty cells with 0
matrix = matrix.astype('int') # make columns to be recognized as integer
```
Here, I assume the first column is the book's name, and the first row is the names of persons. If a person wants to read the book, set the corresponding cell to 1 and otherwise
leave it empty or enter 0. So the table of votes looks like:

| Book | Person_1 | Person_2 |
| ----------- | ----------- | ----------- |
| Book_1 | 1 |  |
| Book_2 | 0 | 1 |

The first line of the above code reads the CSV file. Then, the empty cells are filled with 0, and the values in the columns are cast as integers.

Now let us write a function that gets the list of tasks(here books), including which resources are going to be used by them (here people), the number of cycles(here the number of months), 
and the maximum number of concurrent tasks (here the maximum number of parallel books being read). It returns then the optimum scheduling as a data frame.
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
The code uses a CpModel, then adds constraints to it, and finally uses the CP-SAT solver to find a schedule satisfying the constraints and maximizing the cost function.
The cost function here is defined to maximize resource utilization (here, the number of books read in the book club).

Finally, we can use this function to find the optimum scheduling for our book club.
```python
matrix = matrix.loc[matrix.sum(axis=1) > 1] #remove books that only has a single vote
matrix = matrix.T #transpose the matrix so we have a list of columns containing votes for each book
max_concurrent_tasks = 2 #define maximum number of subgroups that you want to have in a month
num_cycles = 12 #number of months that you want to schedule for
schedule = schedule_tasks(matrix, num_cycles, max_concurrent_tasks)
schedule.to_csv("schedule.csv")
print(schedule)
```
Here, we remove the books that only get a single vote and then transpose the matrix to make each column represent a task (book).
We define the maximum number of subgroups in each month and the number of months we want to plan for, and finally, we use the above-defined function to get the schedule.
We can save the schedule in a CSV file or print it out.
I hope this small script can help you find a better schedule for your book club and enable your club to read more books and engage more people.
