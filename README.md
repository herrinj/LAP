# LAP

This repo contains MATLAB functions and scripts to run examples using the Linearize and Project (LAP) method. 

## References

This code implements the numerical techniques outlined in this paper:

```
@article{Herring2018,
	title = {{LAP}: a Linearize and Project Method for Solving Inverse Problems with Coupled Variables},
	author = {J.~L.~Herring and J.~G.~Nagy and L.~Ruthotto},
	year = {2018},
	note={arXiv preprint arXiv:1705.09992 [math.NA]}
}
```

## Dependencies 

All codes rely heavily on the FAIR toolbox available here: https://github.com/C4IR/FAIR.m. This toolbox is required in the MATLAB filepath to run any of the examples in this repo. 

Codes using hybrid regularization with LAP or block coordinate descent require the IRtools toolbox in the MATLAB filepath, available here: https://github.com/jnagy1/IRtools.

## Acknowledgements 

This work is supported by Emory's University Research Committee and National Science Foundation (NSF) awards DMS 1522760
and DMS 1522599.
