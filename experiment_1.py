import matplotlib.pyplot as plt
import numpy
import time
import subprocess

poolSizes = []
maxPoolSize = 4000
poolSizeStep = 200
epochsPerTest = 10

for i in range(int(maxPoolSize / poolSizeStep), int(maxPoolSize / poolSizeStep) + 1):
    poolSizes.append(i * 100)

configurations = ["BERT_SMALL", "BERT_BASE", "BERT_LARGE"]

csv_file = open("results.csv", "w")

for configuration in configurations:
    data = []

    for i in range(int(maxPoolSize / poolSizeStep), int(maxPoolSize / poolSizeStep) + 1):
        poolSize = i * poolSizeStep
        
        
        with open("logs/{}/{}.log".format(configuration, str(poolSize)), "w") as logfile:
            start_time = time.time_ns()
            
            subprocess.run(
                [
                    "build/SpaceEfficientTransformer." + configuration,
                    "--epochs",
                    str(epochsPerTest),
                    "--optimizations",
                    "on",
                    str(poolSize)
                ],
                stdout=logfile
            )
            
            end_time = time.time_ns()
            time_elapsed = (end_time - start_time) * 1e-9

            data.append(time_elapsed)
            csv_file.write(str(time_elapsed) + ", ")

            print("Progress: Configuration: {}, iteration: {}, time_elapsed: {}".format(configuration, i, time_elapsed))

    csv_file.write("\n")
    plt.plot(poolSizes, data)

plt.ylabel("Run time")
plt.xlabel("Memory pool size")
plt.title("Effect of memory pool size on training time (for 100 iterations)")
plt.legend(configurations)
plt.show()

csv_file.close()