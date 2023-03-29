import math

from ax.utils.notebook.plotting import render
from kerasAxExperiment import ax_client, evaluate

newExperiment = 'optimisedCNNss6Accuracy2.json'

for i in range(100): #run 100 trials
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
    ax_client.save_to_json_file(filepath=newExperiment) #save results after every trial

render(ax_client.get_optimization_trace()) #render optimisation progress to graph
df = ax_client.get_trials_data_frame()

best_parameters, values = ax_client.get_best_parameters()

# the best set of parameters.
for k in best_parameters.items():
  print(k)

# the best score achieved.
means, covariances = values
print(means)


