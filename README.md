# Code for ReSTIR style integrators
Install mitsuba3 `pip install mitsuba`. 

## Emitter only sampling
![Result](emitter_di.png). 

# RIS with WRS 
I used `M=32` and `n=1` with $\hat{p}(x) = \rho(x) L_e(x) G(x)$ according to the paper. 
![Result](wrs_emitter_di.png)

# To test the code
With weighted reservoir sampling with RIS. 
`python wrs_di.py -with_wrs`
With emitter only sampling. 
`python wrs_di.py`