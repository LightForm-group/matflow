
import matplotlib.pyplot as plt
from matflow import utils


def plot_stress_strain(workflow, axis, stresstype="sigma", straintype="epsilon_V^0(F)_vM",
                       datascale="volume_data"):
    """
    Plots Stress agaianst strain for the simulate_volume_element_loading task of a
    given completed matflow workflow.
    
    Parameters
    ----------
    workflow : Competed matflow workflow as dictionary after being passed through load_workflow
        workflow is required to contain 
    axis: Axis to plot stress and strain for. Given as a string. "X", "Y" or "Z".
    stresstype: Type of stress, "sigma", "sigma_vM", etc.
    straintype: Type of strain, "epsilon_U^0(F)", "epsilon_V^0(F)", "epsilon_V^0(F)_vM" etc.
    datascale: "volume_data", "field_data", "grain_data", etc. defaults to "volume_data".
    """
    
    ve_response = workflow.tasks.simulate_volume_element_loading.elements[0].outputs.volume_element_response
    
    tensor_comp = utils.tensor_comps_fromaxis(axis)
    # stress-strain direction is given as component of tensor (i.e X:11, Y:22, Z:33).
    # Python indexes from 0, therefore X:00, Y:11, Z:22
    
    for key in ve_response[datascale].keys():
        if key.find("sigma"):
            stresstype=="sigma" 
        if key.find("epsilon"):
            straintype=="epsilon_V^0(F)_vM"
    
    stress = ve_response[datascale][stresstype]["data"][:, tensor_comp-1, tensor_comp-1]
    
    strain = ve_response[datascale][straintype]["data"]
    
    plt.plot(strain, stress/1e6, linestyle='solid', marker='o', color="k", label='stress-strain', zorder=1)
    plt.xlabel(f"Strain [-]")
    plt.ylabel(f"Stress (MPa)")
    #stress is plotted on scale of MPa.
    