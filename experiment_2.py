import os

configurations = ["BERT_SMALL", "BERT_BASE", "BERT_LARGE"]
# configurations = ["BERT_SMALL"]
timers = [
    "cudaSetup",
    "cudnnExecution",
    "memoryLoadRandom",
    "memoryMakeRoom",
    "memoryManagement",
    "memoryMoveDeviceData",
    "memoryReadFromDevice",
    "memorySpill",
    "memoryUnspill",
    "messages",
    "otherTraining",
    "randomNumberGeneration",
]

# CSV files
for configuration in configurations:
    log_file_names = os.listdir("./logs/" + configuration)
    log_file_names = [int(x[:-4]) for x in log_file_names]
    pool_sizes = sorted(log_file_names)

    
    # CSV Header
    print("Memory pool size, ", end="")
    for pool_size in pool_sizes:
        print(str(pool_size) + ", ", end="")
    print()

    csv_data = []
    for timer in timers:
        csv_data.append([])

    for pool_size in pool_sizes:
        log_file_path = "./logs/{}/{}.log".format(configuration, pool_size)

        with open(log_file_path, "r") as log_file:
            logs = log_file.readlines()

            lines_of_interest = logs[-len(timers):]
            fail_counter = 0
            for line_of_interest in lines_of_interest:
                timer_report = line_of_interest[1:-2]

                timer_components = timer_report.split(": ")
                timer_name = timer_components[0]

                if timer_name not in timers:
                    # print("timer detection problem for file " + log_file_path)
                    csv_data[fail_counter].append("")
                    fail_counter += 1
                else:
                    timer_value = timer_components[1]
                    timer_index_in_csv = timers.index(timer_name)
                    csv_data[timer_index_in_csv].append(timer_value)
    
    for row_index in range(len(timers)):
        print(timers[row_index] + ", ", end="")

        for value in csv_data[row_index]:
            print(value + ", ", end="")
        print()
    
    print()
