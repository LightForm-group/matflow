
import matplotlib.pyplot as plt
from cycler import cycler

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
    
    tensor_comp, unit_vector = utils.tensorcomps_fromaxis(axis)
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
    plt.xlabel(f"Strain $\epsilon$ [-]")
    plt.ylabel(f"Stress $\sigma_{{{tensor_comp}{tensor_comp}}}$ (MPa)") #stress is plotted on scale of MPa (/1e6).
    
    
def plot_stress_latticestrain(true_stress, latticestrain_mean, incs="*", xlim=5000, ylim=None, save="n"):
    """
    Plots true stress against lattice strain for simulate_volume_element_loading
    task of a completed matflow workflow.
    
    Parameters
    ----------
    true_stress: values of true stress as list or array calculated using defdap or from experimental data.
    lattice_strain: Values of lattice strain as 
    """
    
    phase_names = [
             'Ti-alpha',
             'Ti-beta',
             ]
    
    plt.figure()
    plt.title(f"Lattice Strain")
    ax = plt.gca()

    custom_cycler = (cycler(color=[
                                   '#FFE800', '#9FDF00', '#0DCD52', '#00948D', # alpha
                                   '#67001F', '#DF2179', '#CDA0CD' # beta
                                  ]) +
                     cycler(marker=[
                                    's', 'h', '^', 'D', # alpha
                                    '+', 'x', 'P' # beta
                                   ]))
    ax.set_prop_cycle(custom_cycler)

    for phase in phase_names:
        for plane_label, mean_strain in latticestrain_mean[phase].items():
            total_microstrain = ([0] + mean_strain) * 1e6 # scale to micro
            plane_modulus = ((true_stress[1]*1e6) / (total_microstrain[1]*1e-6))/1e9
            print(f"Modulus for plane {plane_label} is: {plane_modulus} ")
            ax.plot(total_microstrain, true_stress, label=plane_label)

    ax.set_xlabel("Lattice Strain ($10^{-6}$)")
    ax.set_xlim([None, xlim])
    ax.set_ylabel("True Stress $\sigma$ (MPa)")
    ax.set_ylim([None, ylim])

    ax.legend()
    if save=='y':
        plt.savefig('microstrain.png', dpi=300)
