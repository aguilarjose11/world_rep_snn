Design of the neural model i am using.

The model i will be using is a model that uses the math found in
literature regarding spiking neural networks. I use the Leaky 
Integrate-and-Fire model as inspiration for this model. 

The potential on the membrane is triggered by inputs from other neurons. 
These inputs will create a spike that takes some time (modified with tau_s) to get to a peak and then has a decay period (modified with tau_m) on which it comes back to zero.

When the potential of the membrane reaches a level v, the membrane is filled with a charge and where the membrane stops being responsive to any inputs. This is refered to time t_i and the potential immediatelly decays down until comming back to some value u_r. We can also measure a time span where the membrane is unresposivne t_r. after that time the membrane can act again. the membrane keeps moving towards u_rest.

* for OULIFSRM, check that the number of inputs is the same as the init_d values. check that the number of layer neurons is the same as init_w