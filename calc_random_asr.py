import random

num_experiments = 100
batch_size = 128
num_classes = 100

total_shared = 0

for _ in range(num_experiments):
    shared_count = 0

    # Generate the first list
    first_list = []
    random_num_1 = random.randint(1, num_classes)
    random_num_2 = random.randint(1, num_classes)

    for _ in range(batch_size // 2):
        first_list.append(random_num_1)
    for _ in range(batch_size // 4):
        first_list.append(random_num_2)
    for _ in range(batch_size // 4):
        first_list.append(random.randint(1, num_classes))

    # Generate the second list
    second_list = [random.randint(1, num_classes) for _ in range(batch_size)]

    # Count the shared numbers
    for num in set(first_list):
        shared_count += min(first_list.count(num), second_list.count(num))

    # Update the total shared count
    total_shared += shared_count / batch_size

# Calculate the average result
average_result = total_shared / num_experiments

# Report the average result
print("Average shared numbers:", average_result)
