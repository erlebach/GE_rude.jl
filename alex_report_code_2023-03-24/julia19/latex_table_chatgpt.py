import matplotlib.pyplot as plt

# Data extracted from the LaTeX table
data = [
    {'value1': 1, 'value2': 1110.1, 'value3': 'a'},
    {'value1': 2, 'value2': 10.1, 'value3': 'b'},
    {'value1': 3, 'value2': 23.113231, 'value3': 'c'},
    {'value1': 4, 'value2': 25.113231, 'value3': 'd'},
]

# Separate data into individual lists
value1 = [d['value1'] for d in data]
value2 = [d['value2'] for d in data]
value3 = [d['value3'] for d in data]

# Create a bar plot for Value 1 and Value 2
fig, ax1 = plt.subplots()
ax1.bar(value1, value2, color='b')
ax1.set_xlabel('Value 1')
ax1.set_ylabel('Value 2', color='b')
ax1.tick_params('y', colors='b')

# Create a second y-axis to display Value 3
ax2 = ax1.twinx()
ax2.plot(value1, value3, 'r-')
ax2.set_ylabel('Value 3', color='r')
ax2.tick_params('y', colors='r')

# Set the plot title and show the plot
plt.title('Plot of Value 1, Value 2, and Value 3')
plt.show()