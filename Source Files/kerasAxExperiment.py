from kerasOptimizerSettings import parameters, keras_mlp_cv_score
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import init_notebook_plotting


ax_client = AxClient.load_from_json_file(filepath='optimisedCNNss6Accuracy2.json') #load existing experiment

# create the client and experiment if they does not exist
#ax_client = AxClient()
"""ax_client.create_experiment( #create the experiment
    name="CNN Optimised Sample_Size_6",
    parameters=parameters,
    objective_name='Validation_Accuracy',
    minimize=True,
)#"""

#Used to get final loss value of a trial with given parameters
def evaluate(parameters):
    return {"Validation_Accuracy": keras_mlp_cv_score(parameters)} #get the optimisation loss score  of a trial

